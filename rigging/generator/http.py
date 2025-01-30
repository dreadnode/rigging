from __future__ import annotations

import asyncio
import base64
import json
import re
import typing as t

import httpx
import jinja2
import jsonpath_ng  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from ruamel.yaml import YAML

from rigging.error import ProcessingError, raise_as
from rigging.generator.base import (
    GeneratedMessage,
    GenerateParams,
    Generator,
    trace_messages,
)
from rigging.message import Message, Role

# TODO:
# - Add request retry mechanics
# - Add request timeout mechanics
# - Add maximum concurrent requests

# Helpers


def _to_str(v: str | dict[str, t.Any]) -> str:
    return json.dumps(v) if isinstance(v, dict) else v


def _to_dict(v: str | dict[str, t.Any]) -> dict[str, t.Any]:
    return t.cast(dict[str, t.Any], json.loads(v)) if isinstance(v, str) else v


def _to_dict_or_str(v: str) -> dict[str, t.Any] | str:
    try:
        return t.cast(dict[str, t.Any], json.loads(v))
    except json.JSONDecodeError:
        return v


# Jinja/template context when building request bodies


class RequestTransformContext(BaseModel):
    """
    Context made available to transforms when building request bodies.

    - In URLs and headers, use `{{ <variable> }}` to inject values.
    - For JSON transforms, use `$<variable>` to inject values.
    - For Jinja transforms, use `{{ <variable> }}` to inject values.
    """

    role: Role
    """Role of the last message in the sequence."""

    content: str
    """Content of the last message in the sequence."""

    all_content: str
    """Concatenation of all message content in the sequence."""

    messages: list[dict[str, t.Any]]
    """List of all messages objects in the sequence (.role, .content)."""

    params: dict[str, t.Any]
    """Merged parameters of the incoming request and generator params."""

    api_key: str
    """API key set on the generator."""

    model: str
    """Model set on the generator."""


# Spec types

InputTransform = t.Literal["json", "jinja"]
OutputTransform = t.Literal["jsonpath", "regex", "jinja"]

TransformT = t.TypeVar("TransformT", InputTransform, OutputTransform)


class TransformStep(BaseModel, t.Generic[TransformT]):
    type: TransformT
    pattern: str | dict[str, t.Any]


class RequestSpec(BaseModel):
    """
    Specifies how to build a request from the messages and context.

    At least one transform is required. It's output will be used as the request body.
    """

    url: str
    method: str = "POST"
    headers: dict[str, str] = {}
    transforms: list[TransformStep[InputTransform]] = Field(min_length=1)


class ResponseSpec(BaseModel):
    """
    Specifies how to validate a response and parse it's body into a generated message.

    The final transform output will be used as the message content.
    """

    valid_status_codes: list[int] = [200]
    transforms: list[TransformStep[OutputTransform]]


class HTTPSpec(BaseModel):
    """Defines how to build requests and parse responses for the HTTPGenerator."""

    request: RequestSpec
    response: ResponseSpec | None = None

    @raise_as(ProcessingError, "Error while transforming input")
    def make_request_body(self, context: RequestTransformContext) -> str:
        result: str = ""

        for transform in self.request.transforms:
            if transform.type == "json":

                def replace_vars(obj: t.Any) -> t.Any:
                    if isinstance(obj, str) and obj.startswith("$"):
                        parts = obj[1:].split(".")
                        val = context.model_dump(mode="json") or {}
                        for part in parts:
                            val = val[part]
                        return val
                    elif isinstance(obj, dict):
                        return {k: replace_vars(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [replace_vars(v) for v in obj]
                    return obj

                result = _to_str(replace_vars(_to_dict(transform.pattern)))

            elif transform.type == "jinja":
                merged_context = context.model_dump(mode="json")
                _result = _to_dict_or_str(result)
                merged_context.update(
                    {
                        # Duplicates here for convenience
                        "result": _result,
                        "data": _result,
                        "output": _result,
                        "body": _result,
                    }
                )

                template = jinja2.Template(_to_str(transform.pattern), undefined=jinja2.StrictUndefined)
                result = template.render(**merged_context)

        if result is None:
            raise Exception("No valid input transform found")

        return result

    @raise_as(ProcessingError, "Error while transforming output")
    def parse_response_body(self, data: str) -> str:
        result = data

        if self.response is None:
            return result

        for transform in self.response.transforms:
            if transform.type == "jinja":
                template = jinja2.Template(_to_str(transform.pattern), undefined=jinja2.StrictUndefined)
                _result = _to_dict_or_str(result)
                result = template.render(
                    # Duplicates here for convenience
                    result=_result,
                    data=_result,
                    output=_result,
                    body=_result,
                )

            elif transform.type == "jsonpath":
                jsonpath_expr = jsonpath_ng.parse(_to_str(transform.pattern))
                if isinstance(result, str):
                    result = json.loads(result)
                matches = [match.value for match in jsonpath_expr.find(result)]
                if len(matches) == 0:
                    raise Exception(f"No matches found for JSONPath: {transform.pattern} from {result}")
                result = json.dumps(matches) if len(matches) > 1 else matches[0]

            elif transform.type == "regex":
                matches = re.findall(_to_str(transform.pattern), result)
                matches = [str(match) for match in matches]
                result = json.dumps(matches) if len(matches) > 1 else matches[0]

        return result

    @raise_as(ProcessingError, "Error while preparing headers")
    def make_headers(self, context: RequestTransformContext) -> dict[str, str]:
        headers = {}
        for key, value in self.request.headers.items():
            template = jinja2.Template(value, undefined=jinja2.StrictUndefined)
            headers[key] = template.render(**context.model_dump(mode="json"))
        return headers

    @raise_as(ProcessingError, "Error while preparing URL")
    def make_url(self, context: RequestTransformContext) -> str:
        template = jinja2.Template(self.request.url, undefined=jinja2.StrictUndefined)
        return template.render(**context.model_dump(mode="json"))


class HTTPGenerator(Generator):
    """
    Generator to map messages to HTTP requests and back.

    The generator takes a `spec` attribute which describes how to encode
    messages into HTTP requests and decode the responses back into messages.

    You can pass this spec as a python dictionary, JSON string, YAML string,
    or a base64 encoded JSON/YAML string.

    ```python
    import rigging as rg

    spec = r\"""
    request:
    url: "https://{{ model }}.crucible.dreadnode.io/submit"
    headers:
        "X-Api-Key": "{{ api_key }}"
        "Content-Type": "application/json"
    transforms:
        - type: "json"
        pattern: {
            "data": "$content"
        }
    response:
    transforms:
        - type: "jsonpath"
        pattern: $.flag,output,message
    \"""

    crucible = rg.get_generator("http!test,api_key=<key>")
    crucible.spec = spec

    chat = await crucible.chat("How about a flag?").run()

    print(chat.conversation)
    ```
    """

    model_config = ConfigDict(validate_assignment=True)

    spec: HTTPSpec | None = None
    """Specification for building/parsing HTTP interactions."""

    @field_validator("spec", mode="before")
    def process_spec(cls, v: t.Any) -> t.Any:
        if not isinstance(v, str):
            return v

        # Check if the string is base64 encoded
        try:
            v = base64.b64decode(v).decode()
        except Exception:
            pass

        # Try to load as JSON
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            pass

        # Try to load as YAML
        try:
            return YAML(typ="safe").load(v)
        except Exception:
            pass

        return v

    @field_serializer("spec", mode="wrap")
    def serialize_spec(self, v: t.Any, _: t.Any) -> t.Any:
        if not isinstance(v, HTTPSpec):
            return v

        return base64.b64encode(v.model_dump_json().encode()).decode()

    async def _generate_message(
        self,
        messages: t.Sequence[Message],
        params: GenerateParams,
    ) -> GeneratedMessage:
        if self.spec is None:
            raise ProcessingError("No spec was provided to the HTTPGenerator")

        # Context for our input transforms
        context = RequestTransformContext(
            role=messages[-1].role,
            content=messages[-1].content,
            all_content="\n".join(m.content for m in messages),
            messages=[m.to_openai_spec() for m in messages],
            params=params.to_dict(),
            api_key=self.api_key,
            model=self.model,
        )

        async with httpx.AsyncClient() as client:
            response = await client.request(
                self.spec.request.method,
                self.spec.make_url(context),
                content=self.spec.make_request_body(context),
                headers=self.spec.make_headers(context),
            )

        content = response.text

        if self.spec.response is not None:
            if response.status_code not in self.spec.response.valid_status_codes:
                raise ProcessingError(f"Received invalid status code: {response.status_code} for {response.url}")

        return GeneratedMessage(
            message=Message(role="assistant", content=self.spec.parse_response_body(content)),
            stop_reason="stop",
            usage=None,
            extra={"status_code": response.status_code, "url": response.url, "headers": response.headers},
        )

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        coros = [self._generate_message(_messages, _params) for _messages, _params in zip(messages, params)]
        generated = await asyncio.gather(*coros)

        for i, (_messages, response) in enumerate(zip(messages, generated)):
            trace_messages(_messages, f"Messages {i+1}/{len(messages)}")
            trace_messages([response], f"Response {i+1}/{len(messages)}")

        return generated
