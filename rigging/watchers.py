"""
Common watcher callback makers for use with generators, chats, and completions.
"""

import json
import typing as t
from pathlib import Path

from loguru import logger

from rigging.data import chats_to_elastic, flatten_chats, s3_object_exists
from rigging.logging import LogLevelLiteral
from rigging.message import ContentText, Message
from rigging.util import shorten_text

if t.TYPE_CHECKING:
    from elasticsearch import AsyncElasticsearch  # type: ignore [import-not-found, unused-ignore]
    from mypy_boto3_s3 import S3Client

    from rigging.chat import Chat, WatchChatCallback


def write_chats_to_jsonl(file: str | Path, *, replace: bool = False) -> "WatchChatCallback":
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

    async def _write_chats_to_jsonl(chats: "list[Chat]") -> None:
        nonlocal replaced

        if file.exists() and replace and not replaced:
            Path.unlink(file)
            replaced = True

        with file.open("a") as f:  # noqa: ASYNC230
            for chat in chats:
                f.write(chat.model_dump_json(exclude_none=True) + "\n")

    return _write_chats_to_jsonl


def write_messages_to_jsonl(file: str | Path, *, replace: bool = False) -> "WatchChatCallback":
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

    async def _write_messages_to_jsonl(chats: "list[Chat]") -> None:
        nonlocal replaced

        if file.exists() and replace and not replaced:
            Path.unlink(file)
            replaced = True

        with file.open("a") as f:  # noqa: ASYNC230
            for chat in flatten_chats(chats):
                f.write(json.dumps(chat) + "\n")

    return _write_messages_to_jsonl


def write_chats_to_elastic(
    client: "AsyncElasticsearch",
    index: str,
    *,
    create_index: bool = True,
    **kwargs: t.Any,
) -> "WatchChatCallback":
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

    async def _write_chats_to_elastic(chats: "list[Chat]") -> None:
        await chats_to_elastic(chats, index, client, create_index=create_index, **kwargs)

    return _write_chats_to_elastic


def write_chats_to_s3(
    client: "S3Client",
    bucket: str,
    key: str,
    *,
    replace: bool = False,
) -> "WatchChatCallback":
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

    async def _write_chats_to_s3(chats: "list[Chat]") -> None:
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


logger_with_color = logger.opt(colors=True)


def _log_message(
    msg: Message,
    *,
    level: LogLevelLiteral = "info",
    max_chars: int | None = None,
    max_lines: int | None = None,
) -> None:
    if msg.role == "tool":  # Tools get less space
        max_chars = max_chars // 2 if max_chars else None
        max_lines = max_lines // 2 if max_lines else None

    def shorten(text: str) -> str:
        return shorten_text(text, max_chars=max_chars, max_lines=max_lines)

    role_style = (
        "magenta"
        if msg.role == "system"
        else "blue"
        if msg.role == "user"
        else "magenta"
        if msg.role == "tool"
        else "cyan"
    )
    text_style = "white" if msg.role == "assistant" else "dim"

    statement = f"\n<{role_style}>[{msg.role}]:</>\n"
    statement_kwargs: dict[str, str] = {}

    if len(msg.content_parts) == 1 and isinstance(msg.content_parts[0], ContentText):
        if content := msg.content_parts[0].text.strip():
            statement += f"<{text_style}>{{content}}</>\n"
            statement_kwargs["content"] = shorten(content)

    else:
        for i, part in enumerate(msg.content_parts):
            if content_str := str(part):
                statement += f" |- <{text_style}>{{content{i}}}</>\n"
                statement_kwargs[f"content{i}"] = shorten(content_str)

    for i, tool_call in enumerate(msg.tool_calls or []):
        function_name = tool_call.function.name
        statement += f" |- <magenta>{function_name}</>(<dim>{{args{i}}}</>)\n"
        statement_kwargs[f"args{i}"] = shorten_text(
            tool_call.function.arguments,
            max_chars=max_chars // 4 if max_chars else None,  # Tool args get less space
        )

    logger_with_color.log(level.upper(), statement, **statement_kwargs)


def make_stream_to_logs(
    *,
    level: LogLevelLiteral = "info",
    max_chars: int = 500,
    max_lines: int = 20,
) -> "WatchChatCallback":
    """
    Creates a watch-style callback that streams conversation progress to the console (via Loguru).

    This version intelligently tracks the conversation by comparing the hash history
    of messages, making it robust against unpredictable UUIDs or history rewrites.
    If the history deviates, it treats it as a new conversation.

    Args:
        level: The Loguru log level to use.
        max_chars: Max characters to display for content before shortening.
        max_lines: Max lines to display for content before shortening.

    Returns:
        An awaitable callback function for use with `ChatPipeline.watch()`.
    """
    last_tracked_hashes: list[int] = []
    last_printed_hash: int | None = None
    error_logged_for_hash: int | None = None

    logger.enable("rigging")

    async def stream_to_logs(chats: "list[Chat]") -> None:
        nonlocal last_tracked_hashes, last_printed_hash, error_logged_for_hash

        if not chats:
            return

        chat = chats[0]

        current_hashes = [m.hash for m in chat.all]
        is_continuation = (
            last_tracked_hashes
            and current_hashes[: len(last_tracked_hashes)] == last_tracked_hashes
        )

        print_from_index = 0

        if not is_continuation:
            last_printed_hash = None

        elif last_printed_hash:
            # It IS a continuation. Find where we left off.
            try:
                last_printed_index = current_hashes.index(last_printed_hash)
                print_from_index = last_printed_index + 1
            except ValueError:
                # This can happen if the history was rewritten in a way that
                # preserved the beginning but removed the last message we saw.
                # We can try to find the last valid message from our old history.
                for i in range(len(last_tracked_hashes) - 1, -1, -1):
                    try:
                        last_known_index = current_hashes.index(last_tracked_hashes[i])
                        print_from_index = last_known_index + 1
                        break
                    except ValueError:
                        continue
                else:
                    print_from_index = 0

        for i in range(print_from_index, len(chat.all)):
            message = chat.all[i]
            current_hash = current_hashes[i]
            if current_hash == last_printed_hash:
                continue

            _log_message(
                message,
                level=level,
                max_chars=max_chars,
                max_lines=max_lines,
            )

            last_printed_hash = current_hash

        last_tracked_hashes = current_hashes

    return stream_to_logs


stream_to_logs = make_stream_to_logs()
"""
A convenience watch callback for chat pipelines to stream conversation messages to the console (via Loguru).

Use `make_stream_to_logs()` to create a custom version with specific parameters.

Example:
    ```python
    chat = (
        await generator.chat(...)
        .watch(stream_to_logs)
        .run()
    )
    ```
"""
