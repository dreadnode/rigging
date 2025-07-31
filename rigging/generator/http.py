import asyncio
import base64
import contextlib
import json
import re
import typing as t

import httpx
import jinja2
import jsonpath_ng  # type: ignore [import-untyped]
import typing_extensions as te
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

# TODO: Look at:
# - Add request retry mechanics
# - Add request timeout mechanics
# - Add maximum concurrent requests

DEFAULT_MAX_RETRIES = 5

# Helpers


def _to_str(v: str | dict[str, t.Any]) -> str:
    return json.dumps(v) if isinstance(v, dict) else v


def _to_dict(v: str | dict[str, t.Any]) -> dict[str, t.Any]:
    return t.cast("dict[str, t.Any]", json.loads(v)) if isinstance(v, str) else v


def _to_dict_or_str(v: str) -> dict[str, t.Any] | str:
    try:
        return t.cast("dict[str, t.Any]", json.loads(v))
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

    state: dict[str, t.Any] = Field(default_factory=dict)
    """Mutable dictionary for dynamic state like access tokens to use in your spec."""


# Spec types

InputTransform = t.Literal["json", "jinja"]
OutputTransform = t.Literal["jsonpath", "regex", "jinja"]

TransformT = t.TypeVar("TransformT", InputTransform, OutputTransform)


class TransformStep(BaseModel, t.Generic[TransformT]):
    type: TransformT
    """Type of transform to apply."""

    pattern: str | dict[str, t.Any]
    """Pattern to use for the transform."""


class RequestSpec(BaseModel):
    """
    Specifies how to build a request from the messages and context.

    At least one transform is required. It's output will be used as the request body.
    """

    url: str
    """URL to send the request to (Jinja templates supported)."""

    method: str = "POST"
    """HTTP method to use for the request."""

    headers: dict[str, str] = {}
    """Headers to include in the request (Jinja templates supported)."""

    timeout: int | None = None
    """Timeout in seconds for the request."""

    transforms: list[TransformStep[InputTransform]] = Field(min_length=1)
    """Transforms to apply to the messages to build the request body."""


class ResponseSpec(BaseModel):
    """
    Specifies how to validate a response and parse it's body into a generated message.

    The final transform output will be used as the message content.
    """

    valid_status_codes: list[int] = [200]
    """Valid status codes for the response."""

    transforms: list[TransformStep[OutputTransform]]
    """Transforms to apply to the response body to generate the message content."""


class HTTPSpec(BaseModel):
    """Defines how to build requests and parse responses for the HTTPGenerator."""

    request: RequestSpec
    """Specification for building the request."""

    response: ResponseSpec | None = None
    """Specification for parsing the response."""

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
                    if isinstance(obj, dict):
                        return {k: replace_vars(v) for k, v in obj.items()}
                    if isinstance(obj, list):
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
                    },
                )

                template = jinja2.Template(
                    _to_str(transform.pattern),
                    undefined=jinja2.StrictUndefined,
                )
                result = template.render(**merged_context)

        if result is None:
            raise RuntimeError("No valid input transform found")

        return result

    @raise_as(ProcessingError, "Error while transforming output")
    def parse_response_body(self, data: str) -> str:
        result = data

        if self.response is None:
            return result

        for transform in self.response.transforms:
            if transform.type == "jinja":
                template = jinja2.Template(
                    _to_str(transform.pattern),
                    undefined=jinja2.StrictUndefined,
                )
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
                    raise RuntimeError(
                        f"No matches found for JSONPath: {transform.pattern} from {result}",
                    )
                if len(matches) == 1:
                    matches = matches[0]
                result = matches if isinstance(matches, str) else json.dumps(matches)

            elif transform.type == "regex":
                matches = re.findall(_to_str(transform.pattern), result)
                matches = [str(match) for match in matches]
                result = json.dumps(matches) if len(matches) > 1 else str(matches[0])

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


class HttpAuthConfigDict(te.TypedDict):
    """Configuration for API key authentication."""

    header: str
    """The name of the header, e.g., 'Authorization' or 'X-Api-Key'."""
    format: te.NotRequired[str]
    """
    The format string for the header's value. Defaults to '{api_key}'.
    Example: 'Bearer {api_key}'
    """


class ApiResponseConfigDict(te.TypedDict):
    """Defines how to parse content from an API response."""

    content_path: te.NotRequired[str]
    """
    JSONPath to extract the primary message content from a successful response.
    Defaults to '$' to return the entire response body.
    Example: '$.choices[0].message.content'
    """
    error_path: te.NotRequired[str]
    """
    JSONPath to extract a detailed error message if the response is unsuccessful.
    If not found, the full response body will be used.
    Example: '$.error.message'
    """


class HttpAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    header: str
    """The name of the header, e.g., 'Authorization' or 'X-Api-Key'."""
    format: str = "{api_key}"
    """
    The format string for the header's value. Defaults to '{api_key}'.
    Example: 'Bearer {api_key}'
    """


class ApiResponseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_path: str = "$"
    """
    JSONPath to extract the primary message content from a successful response.
    Defaults to '$' to return the entire response body.
    Example: '$.choices[0].message.content'
    """
    error_path: str | None = None
    """
    JSONPath to extract a detailed error message if the response is unsuccessful.
    If not found, the full response body will be used.
    Example: '$.error.message'
    """


HttpHookAction = t.Literal["retry", "raise", "continue"]
HttpHook = t.Callable[["HTTPGenerator", httpx.Response], t.Awaitable[HttpHookAction | None]]
"""
Hook to run after each HTTP request of the HTTPGenerator.

The hook receives the generator instance and the HTTP response.

It can return:
- "retry": to retry the request.
- "raise": to raise an error.
- "continue"/None: to continue processing without retrying.
"""


class HTTPGenerator(Generator):
    """
    Generator to map messages to HTTP requests and back.

    The generator takes a `spec` attribute which describes how to encode
    messages into HTTP requests and decode the responses back into messages.

    You can pass this spec as a python dictionary, JSON string, YAML string,
    or a base64 encoded JSON/YAML string.

    Example:
        ```
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
    state: dict[str, t.Any] = Field(default_factory=dict)
    """Mutable dictionary for dynamic state like access tokens to use in your spec."""
    hook: HttpHook | None = Field(default=None, exclude=True)
    """Optional hook to run after each HTTP request with the option to retry or raise an error."""
    max_retries: int = DEFAULT_MAX_RETRIES
    """"Maximum number of retries the hook can trigger. Defaults to 5."""

    @field_validator("spec", mode="before")
    @classmethod
    def process_spec(cls, v: t.Any) -> t.Any:
        if not isinstance(v, str):
            return v

        # Check if the string is base64 encoded
        with contextlib.suppress(Exception):
            v = base64.b64decode(v).decode()

        # Try to load as JSON
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(v)

        # Try to load as YAML
        with contextlib.suppress(Exception):
            return YAML(typ="safe").load(v)

        return v

    @field_serializer("spec", mode="wrap")
    def serialize_spec(self, v: t.Any, _: t.Any) -> t.Any:
        if not isinstance(v, HTTPSpec):
            return v

        return base64.b64encode(v.model_dump_json().encode()).decode()

    @classmethod
    def for_json_endpoint(
        cls,
        url: str,
        request: dict[str, t.Any],  # Renamed for clarity
        model: str | None = None,
        api_key: str | None = None,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        auth: HttpAuthConfigDict | HttpAuthConfig | None = None,
        response: ApiResponseConfigDict | ApiResponseConfig | None = None,
        valid_status_codes: list[int] | None = None,
        timeout: int | None = None,
        hook: HttpHook | None = None,
        state: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> "HTTPGenerator":
        """
        Creates an HTTPGenerator from a simplified, high-level API definition for JSON endpoints.

        This is the recommended entry point for most use cases. It provides full
        autocompletion when creating configuration dictionaries in your IDE.

        Example:
            ```
            import rigging as rg

            openai_api = rg.HTTPGenerator.for_json_endpoint(
                "https://api.openai.com/v1/chat/completions",
                auth={
                    "header": "Authorization",
                    "format": "Bearer {api_key}"
                },
                request={
                    "model": "{{ model }}",
                    "messages": "$messages",
                },
                response={
                    "content_path": "$.choices[0].message.content",
                    "error_path": "$.error.message"
                }
            )
            ```

        Args:
            url: The URL of the API endpoint (supports Jinja templates).
            request: A dictionary defining the request body structure.
                Use `$<variable>` to reference context variables.
            model: Optional model name for the generator.
            api_key: Optional API key to use for authentication.
            method: HTTP method to use (default is "POST").
            headers: Optional headers to include in the request.
                Defaults to "Content-Type": "application/json".
            auth: Optional authentication configuration for API key headers.
            response: Optional configuration for parsing the response body.
            valid_status_codes: List of valid HTTP status codes (default is [200]).
            timeout: Optional timeout in seconds for the request.
            hook: Optional hook to run after each HTTP request.
            state: Optional mutable dictionary for dynamic state like access tokens.
            **kwargs: Additional keyword arguments passed to the generator.

        Returns:
            An instance of HTTPGenerator configured for the specified endpoint.
        """
        auth_model = HttpAuthConfig.model_validate(auth) if auth else None
        response_model = ApiResponseConfig.model_validate(response or {})

        final_headers = (headers or {"Content-Type": "application/json"}).copy()
        if auth_model:
            jinja_auth_format = auth_model.format.replace("{api_key}", "{{ api_key }}")
            final_headers[auth_model.header] = jinja_auth_format

        request_spec = RequestSpec(
            url=url,
            method=method,
            headers=final_headers,
            timeout=timeout,
            transforms=[TransformStep[InputTransform](type="json", pattern=request)],
        )

        response_transforms = []
        if response_model.content_path != "$":
            response_transforms.append(
                TransformStep[OutputTransform](type="jsonpath", pattern=response_model.content_path)
            )

        response_spec = ResponseSpec(
            valid_status_codes=valid_status_codes or [200],
            transforms=response_transforms,
        )

        spec = HTTPSpec(request=request_spec, response=response_spec)
        return cls(model=model, api_key=api_key, spec=spec, hook=hook, state=state, **kwargs)

    @classmethod
    def for_text_endpoint(
        cls,
        url: str,
        request: str,
        response_pattern: str | None = None,
        response_pattern_type: t.Literal["regex", "jinja"] = "regex",
        model: str | None = None,
        api_key: str | None = None,
        method: str = "POST",
        headers: dict[str, str] | None = None,
        auth: HttpAuthConfigDict | HttpAuthConfig | None = None,
        valid_status_codes: list[int] | None = None,
        timeout: int | None = None,
        hook: HttpHook | None = None,
        state: dict[str, t.Any] | None = None,
        **kwargs: t.Any,
    ) -> "HTTPGenerator":
        """
        Creates an HTTPGenerator from a template-based definition.

        Ideal for simpler text-based APIs where the request body is generated
        from a Jinja2 template and the response is parsed with a Regex or another template.

        Example:
            ```
            import rigging as rg

            text_api = rg.HTTPGenerator.for_text_endpoint(
                "http://api.example.com/prompt",
                "User prompt: {{ content }}", # Jinja template
                response_pattern="Response: (.*)", # Regex to extract content
                auth={
                    "header": "Authorization",
                    "format": "Bearer {api_key}"
                }
            )
            ```

        Args:
            url: The URL of the API endpoint (supports Jinja templates).
            request: A Jinja template string for the request body.
            response_pattern: Optional pattern to extract content from the response.
                If not provided, the entire response body will be used.
            response_pattern_type: Type of the response pattern, either "regex" or "jinja
            model: Optional model name for the generator.
            api_key: Optional API key to use for authentication.
            method: HTTP method to use (default is "POST").
            headers: Optional headers to include in the request.
                Defaults to "Content-Type": "text/plain".
            auth: Optional authentication configuration for API key headers.
            valid_status_codes: List of valid HTTP status codes (default is [200]).
            timeout: Optional timeout in seconds for the request.
            hook: Optional hook to run after each HTTP request.
            state: Optional mutable dictionary for dynamic state like access tokens.
            **kwargs: Additional keyword arguments passed to the generator.
        """
        auth_model = HttpAuthConfig.model_validate(auth) if auth else None

        final_headers = (headers or {"Content-Type": "text/plain"}).copy()
        if auth_model:
            jinja_auth_format = auth_model.format.replace("{api_key}", "{{ api_key }}")
            final_headers[auth_model.header] = jinja_auth_format

        request_spec = RequestSpec(
            url=url,
            method=method,
            headers=final_headers,
            timeout=timeout,
            transforms=[TransformStep[InputTransform](type="jinja", pattern=request)],
        )

        response_transforms = []
        if response_pattern:
            response_transforms.append(
                TransformStep[OutputTransform](
                    type=response_pattern_type,
                    pattern=response_pattern,
                )
            )

        response_spec = ResponseSpec(
            valid_status_codes=valid_status_codes or [200],
            transforms=response_transforms,
        )

        spec = HTTPSpec(request=request_spec, response=response_spec)
        return cls(model=model, api_key=api_key, spec=spec, hook=hook, state=state, **kwargs)

    async def _generate_message(
        self,
        messages: t.Sequence[Message],
        params: GenerateParams,
    ) -> GeneratedMessage:
        if self.spec is None:
            raise ProcessingError("No spec was provided to the HTTPGenerator")

        response: httpx.Response | None = None
        for _ in range(self.max_retries):
            # Context for our input transforms
            context = RequestTransformContext(
                role=messages[-1].role,
                content=messages[-1].content,
                all_content="\n".join(m.content for m in messages),
                messages=[m.to_openai() for m in messages],
                params=params.to_dict(),
                api_key=self.api_key or "",
                model=self.model,
                state=self.state,
            )

            # Conditionally set the timeout to avoid overriding the default "unset" value
            kwargs: dict[str, t.Any] = {}
            if self.spec.request.timeout is not None:
                kwargs["timeout"] = self.spec.request.timeout
            elif params.timeout is not None:
                kwargs["timeout"] = params.timeout
            elif self.params.timeout is not None:
                kwargs["timeout"] = self.params.timeout

            async with httpx.AsyncClient() as client:
                response = await client.request(
                    self.spec.request.method,
                    self.spec.make_url(context),
                    content=self.spec.make_request_body(context),
                    headers=self.spec.make_headers(context),
                    **kwargs,
                )

            if self.hook:
                action = await self.hook(self, response)
                if action == "retry":
                    continue
                if action == "raise":
                    raise ProcessingError(
                        f"Hook instructed to raise an error for status {response.status_code}. Response: {response.text}"
                    )

            content = response.text

            if (
                self.spec.response is not None
                and response.status_code not in self.spec.response.valid_status_codes
            ):
                raise ProcessingError(
                    f"Received invalid status code: {response.status_code} for {response.url}",
                )

            return GeneratedMessage(
                message=Message(role="assistant", content=self.spec.parse_response_body(content)),
                stop_reason="stop",
                usage=None,
                extra={
                    "status_code": response.status_code,
                    "url": response.url,
                    "headers": response.headers,
                },
            )

        raise ProcessingError(
            f"Request failed after {self.max_retries} retry attempts. "
            f"Final status: {response.status_code if response else 'unk'}. "
            f"Response: {response.text if response else 'unk'}"
        )

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        coros = [
            self._generate_message(_messages, _params)
            for _messages, _params in zip(messages, params, strict=True)
        ]
        generated = await asyncio.gather(*coros)

        for i, (_messages, response) in enumerate(zip(messages, generated, strict=True)):
            trace_messages(_messages, f"Messages {i + 1}/{len(messages)}")
            trace_messages([response], f"Response {i + 1}/{len(messages)}")

        return generated
