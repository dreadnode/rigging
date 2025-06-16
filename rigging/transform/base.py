import typing as t

from rigging.generator import GenerateParams
from rigging.message import (
    Message,
)

if t.TYPE_CHECKING:
    from rigging.chat import Chat


@t.runtime_checkable
class PostTransform(t.Protocol):
    def __call__(
        self,
        chat: "Chat",
        /,
    ) -> "t.Awaitable[Chat]":
        """
        Passed messages and params to transform.
        """
        ...


@t.runtime_checkable
class Transform(t.Protocol):
    def __call__(
        self,
        messages: list[Message],
        params: GenerateParams,
        /,
    ) -> t.Awaitable[tuple[list[Message], GenerateParams, PostTransform | None]]:
        """
        Passed messages and params to transform.

        May return an optional post-transform callback to be executed to unwind the transformation.
        """
        ...
