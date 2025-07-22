import typing as t

from loguru import logger

if t.TYPE_CHECKING:
    from rigging.message import Message

CacheMode = t.Literal["latest"]
"""
How to handle cache_control entries on messages.

- latest: Assign cache_control to the latest 2 non-assistant messages in the pipeline before inference.
"""


def apply_cache_mode_to_messages(
    mode: CacheMode | None,
    messages: "list[list[Message]]",
) -> "list[list[Message]]":
    if mode is None:
        return messages

    if mode != "latest":
        logger.warning(
            f"Unknown caching mode '{mode}', defaulting to 'latest'",
        )
        mode = "latest"

    # first remove existing cache settings
    updated: list[list[Message]] = []
    for _messages in messages:
        updated = [
            *updated,
            [m.clone().cache(cache_control=False) for m in _messages],
        ]

    # then apply the latest cache settings
    for _messages in updated:
        for message in [m for m in _messages if m.role != "assistant"][-2:]:
            message.cache(cache_control=True)

    return updated
