import abc
import typing as t

import litellm  # type: ignore
from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator

from rigging.chat import PendingChat
from rigging.error import InvalidModelSpecifiedError
from rigging.message import (
    Message,
    MessageDict,
)

# We should probably let people configure
# this independently, but for now we'll
# fix it to prevent confusion
litellm.drop_params = True


# TODO: Ideally we flex this to support arbitrary
# generator params, but we'll limit things
# for now until we understand the use cases
#
# TODO: We also would like to support N-style
# parallel generation eventually -> need to
# update our interfaces to support that
class GenerateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    api_base: str | None = None
    timeout: int | None = None
    seed: int | None = None

    @field_validator("stop", mode="before")
    def validate_stop(cls, value: t.Any) -> t.Any:
        if isinstance(value, str):
            return value.split(";")
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            return value
        raise ValueError("Stop sequences must be a list or a string separated by ';'")


class Generator(BaseModel, abc.ABC):
    model: str
    api_key: str | None = None
    params: GenerateParams

    def _merge_params(self, overloads: GenerateParams) -> dict[str, t.Any]:
        params: dict[str, t.Any] = self.params.model_dump(exclude_unset=True) if self.params else {}
        for name, value in overloads.model_dump(exclude_unset=True).items():
            if value is not None:
                params[name] = value
        return params

    @abc.abstractmethod
    def complete(self, messages: t.Sequence[Message], overloads: GenerateParams) -> Message:
        ...

    @t.overload
    def chat(self, messages: t.Sequence[MessageDict], overloads: GenerateParams | None = None) -> PendingChat:
        ...

    @t.overload
    def chat(self, messages: t.Sequence[Message], overloads: GenerateParams | None = None) -> PendingChat:
        ...

    def chat(
        self, messages: t.Sequence[MessageDict] | t.Sequence[Message], overloads: GenerateParams | None = None
    ) -> PendingChat:
        return PendingChat(self, Message.fit_list(messages), overloads or GenerateParams())


class LiteLLMGenerator(Generator):
    def complete(self, messages: t.Sequence[Message], overloads: GenerateParams = GenerateParams()) -> Message:
        logger.trace("--- Conversation ---")
        logger.trace("\n".join([str(msg) for msg in messages]))
        logger.trace("---")

        messages_as_dicts = [message.model_dump() for message in messages]
        complete_params = self._merge_params(overloads)
        result = litellm.completion(self.model, messages_as_dicts, api_key=self.api_key, **complete_params)
        response = result.choices[-1].message.content.strip()
        next_message = Message(role="assistant", content=response)

        logger.trace("--- Response ---")
        logger.trace(str(next_message))
        logger.trace("---")

        return next_message


def get_generator(identifier: str) -> Generator:
    """
    Get a generator by an identifier string. Uses LiteLLM by default.

    <provider>!<model>,<**kwargs>

    (provider is optional and defaults to "litellm" if not specified)

    :param identifier: The identifier string to use to get a generator
    :return: The generator

    :raises InvalidModelSpecified: If the identifier is invalid

    Examples:
        "gpt-3.5-turbo" -> LiteLLMGenerator(model="gpt-3.5-turbo")
        "litellm!claude-2.1" -> LiteLLMGenerator(model="claude-2.1")
        "mistral/mistral-tiny" -> LiteLLMGenerator(model="mistral/mistral-tiny")

    You can also specify arguments to the generator by comma-separating them#
        "mistral/mistral-medium,max_tokens=1024"
        "gpt-4-0613,temperature=0.9,max_tokens=512"
        "claude-2.1,stop_sequences=Human:;test,max_tokens=100"

        (These get parsed as GenerateParams)
    """

    provider: str = "litellm"
    model: str = identifier
    api_key: str | None = None
    params: GenerateParams = GenerateParams()

    # Split provider, model, and kwargs

    try:
        if "!" in identifier:
            provider, model = identifier.split("!")

        if "," in model:
            model, kwargs_str = model.split(",", 1)
            kwargs = dict(arg.split("=") for arg in kwargs_str.split(","))
            api_key = kwargs.pop("api_key", None)
            params = GenerateParams(**kwargs)
    except Exception as e:
        raise InvalidModelSpecifiedError(identifier) from e

    if provider == "litellm":
        return LiteLLMGenerator(model=model, api_key=api_key, params=params)
    else:
        raise InvalidModelSpecifiedError(identifier)
