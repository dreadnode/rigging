"""
Common watcher callback makers for use with generators, chats, and completions.
"""

from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

from rigging.data import chats_to_elastic, flatten_chats, s3_object_exists

if t.TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch
    from mypy_boto3_s3 import S3Client

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


def write_chats_to_s3(client: S3Client, bucket: str, key: str, replace: bool = False) -> WatchChatCallback:
    """
    Create a watcher to write each chat to an Amazon S3 bucket.

    Args:
        client: The S3 client to use.
        bucket: The bucket to write to.
        key: The key to write to.
        replace: If the file should be replaced if it already exists.

    Returns:
        A callback to use in [rigging.chat.ChatPipeline.watch][]
        or [rigging.generator.Generator.watch][].
    """

    replaced: bool = False

    async def _write_chats_to_s3(chats: list[Chat]) -> None:
        nonlocal replaced

        content: str = ""

        if await s3_object_exists(client, bucket, key):
            if replace and not replaced:
                # if the object exists, we want to replace it and has not been replaced yet, delete it
                client.delete_object(Bucket=bucket, Key=key)
                replaced = True

            else:
                # if we're not replacing or we have already replaced, read the existing object
                response = client.get_object(Bucket=bucket, Key=key)
                content = response["Body"].read().decode("utf-8")

        # append the new chats to the existing content
        for chat in chats:
            content += chat.model_dump_json() + "\n"

        # write the new content to the object
        client.put_object(Bucket=bucket, Key=key, Body=content)

    return _write_chats_to_s3
