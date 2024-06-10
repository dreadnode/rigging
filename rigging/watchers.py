"""
Common watcher callback makers for use with generators, chats, and completions.
"""

from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

from rigging.data import chats_to_elastic, flatten_chats

if t.TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch

    from rigging.chat import Chat, WatchChatCallback


def write_chats_to_jsonl(file: str | Path, *, replace: bool = False) -> WatchChatCallback:
    """
    Create a watcher to write each chat as a single JSON line appended to a file.

    Args:
        file: The file to write to.
        replace: If the file should be replaced if it already exists.

    Returns:
        A callback to use in [rigging.chat.ChatPipeline.watch][]
        or [rigging.generator.Generator.watch][].
    """

    # We could do the file delete externally, but I don't
    # want to produce a side effect of simply creating the
    # callback.

    file = Path(file)
    replaced: bool = False

    async def _write_chats_to_jsonl(chats: list[Chat]) -> None:
        nonlocal replaced

        if file.exists() and replace and not replaced:
            os.remove(file)
            replaced = True

        with file.open("a") as f:
            for chat in chats:
                f.write(chat.model_dump_json() + "\n")

    return _write_chats_to_jsonl


def write_messages_to_jsonl(file: str | Path, *, replace: bool = False) -> WatchChatCallback:
    """
    Create a watcher to flatten chats to individual messages (like Dataframes) and append to a file.

    Args:
        file: The file to write to.
        replace: If the file should be replaced if it already exists.

    Returns:
        A callback to use in [rigging.chat.ChatPipeline.watch][]
        or [rigging.generator.Generator.watch][].
    """
    file = Path(file)
    replaced: bool = False

    async def _write_messages_to_jsonl(chats: list[Chat]) -> None:
        nonlocal replaced

        if file.exists() and replace and not replaced:
            os.remove(file)
            replaced = True

        with file.open("a") as f:
            for chat in flatten_chats(chats):
                f.write(json.dumps(chat) + "\n")

    return _write_messages_to_jsonl


def write_chats_to_elastic(
    client: AsyncElasticsearch, index: str, *, create_index: bool = True, **kwargs: t.Any
) -> WatchChatCallback:
    """
    Create a watcher to write each chat to an ElasticSearch index.

    Args:
        client: The AsyncElasticSearch client to use.
        index: The index to write to.
        create_index: Whether to create the index if it doesn't exist and update its mapping.
        kwargs: Additional keyword arguments to be passed to the Elasticsearch client.

    Returns:
        A callback to use in [rigging.chat.ChatPipeline.watch][]
        or [rigging.generator.Generator.watch][].
    """

    async def _write_chats_to_elastic(chats: list[Chat]) -> None:
        await chats_to_elastic(chats, index, client, create_index=create_index, **kwargs)

    return _write_chats_to_elastic
