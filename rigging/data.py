"""
Utilities for converting chat data between different formats.
"""

import itertools
import json
import typing as t

import pandas as pd
from mypy_boto3_s3 import S3Client

from rigging.chat import Chat
from rigging.message import Message

if t.TYPE_CHECKING:
    import elasticsearch
    from elastic_transport import ObjectApiResponse


def flatten_chats(chats: Chat | t.Sequence[Chat]) -> list[dict[t.Any, t.Any]]:
    """
    Flatten a list of chats into a individual messages with duplicated
    properties relevant to the chat.

    Args:
        chats: A Chat or list of Chat objects.

    Returns:
        A list of flat Message objects as dictionaries.
    """
    chats = [chats] if isinstance(chats, Chat) else chats

    flattened: list[dict[t.Any, t.Any]] = []
    for chat in chats:
        generator_id = chat.generator_id

        # We let pydantic do the heavy lifting here
        chat_json = chat.model_dump(
            include={"uuid", "timestamp", "metadata", "usage", "extra"},
            mode="json",
        )
        metadata = chat_json.pop("metadata")
        usage = chat_json.pop("usage")
        extra = chat_json.pop("extra")

        generated = False
        for messages in [chat.messages, chat.generated]:
            for message in messages:
                message_dict = message.model_dump(mode="json")
                message_id = message_dict.pop("uuid")
                flattened.append(
                    {
                        "chat_id": chat_json["uuid"],
                        "chat_metadata": metadata,
                        "chat_generator_id": generator_id,
                        "chat_timestamp": chat_json["timestamp"],
                        "chat_stop_reason": chat.stop_reason,
                        "chat_usage": usage,
                        "chat_extra": extra,
                        "generated": generated,
                        "message_id": message_id,
                        **message_dict,
                    },
                )
            generated = True

    return flattened


def unflatten_chats(messages: t.Sequence[dict[t.Any, t.Any]]) -> list[Chat]:
    """
    Unflatten a list of messages into a list of Chat objects.

    Args:
        messages: A list of flat Message objects in the format from [rigging.data.flatten_chats][].

    Returns:
        A list of Chat objects.
    """

    def by_chat_id(message: dict[t.Any, t.Any]) -> t.Any:
        return message["chat_id"]

    sorted_messages = sorted(messages, key=by_chat_id)
    grouped_by = itertools.groupby(sorted_messages, key=by_chat_id)

    chats = []
    for chat_id, chat_messages in grouped_by:
        _messages = []
        _generated = []
        _first_message: dict[t.Any, t.Any] = {}

        for message_data in chat_messages:
            if not _first_message:
                _first_message = message_data

            message = Message(
                role=message_data["role"],
                content=message_data["content"],
                uuid=message_data["message_id"],
            )
            if message_data["generated"]:
                _generated.append(message)
            else:
                _messages.append(message)

        if not _first_message:
            raise ValueError("Grouped messages yieled an empty chat")

        chat = Chat(
            uuid=chat_id,
            timestamp=_first_message["chat_timestamp"],
            messages=_messages,
            generated=_generated,
            metadata=json.loads(_first_message["chat_metadata"]),
            stop_reason=_first_message["chat_stop_reason"],
            usage=json.loads(_first_message["chat_usage"]),
            extra=json.loads(_first_message["chat_extra"]),
            generator_id=_first_message["chat_generator_id"],
        )
        chats.append(chat)

    return chats


# Pandas


def chats_to_df(chats: Chat | t.Sequence[Chat]) -> pd.DataFrame:
    """
    Convert a Chat or list of Chat objects into a pandas DataFrame.

    Note:
        The messages will be flatted and can be joined by the
        chat_id column.

    Args:
        chats: A Chat or list of Chat objects.

    Returns:
        A pandas DataFrame containing the chat data.

    """
    chats = [chats] if isinstance(chats, Chat) else chats

    flattened = flatten_chats(chats)

    # TODO: Come back to indexing

    return pd.DataFrame(flattened).astype(
        {
            "chat_id": "string",
            "chat_metadata": "string",
            "chat_generator_id": "string",
            "chat_timestamp": "datetime64[ms]",
            "chat_stop_reason": "string",
            "chat_usage": "string",
            "chat_extra": "string",
            "generated": "bool",
            "message_id": "string",
            "role": "category",
            "content": "string",
            "parts": "string",
        },
    )


def df_to_chats(df: pd.DataFrame) -> list[Chat]:
    """
    Convert a pandas DataFrame into a list of Chat objects.

    Note:
        The DataFrame should have the same structure as the one
        generated by the `chats_to_df` function.

    Args:
        df: A pandas DataFrame containing the chat data.

    Returns:
        A list of Chat objects.

    """
    chats = []
    for chat_id, chat_group in df.groupby("chat_id"):
        chat_data = chat_group.iloc[0]
        messages = []
        generated = []

        for _, message_data in chat_group.iterrows():
            message = Message(
                role=message_data["role"],
                content=message_data["content"],
                uuid=message_data["message_id"],
                # TODO: I don't believe this is safe to deserialize
                # here as we aren't bonded to the underlying rg.Model
                # which was the original object. Skipping for now.
                # parts=json.loads(message_data["parts"]),
            )
            if message_data["generated"]:
                generated.append(message)
            else:
                messages.append(message)

        chat = Chat(
            uuid=chat_id,
            timestamp=chat_data["chat_timestamp"],
            messages=messages,
            generated=generated,
            metadata=json.loads(chat_data["chat_metadata"]),
            stop_reason=chat_data["chat_stop_reason"],
            usage=json.loads(chat_data["chat_usage"]),
            extra=json.loads(chat_data["chat_extra"]),
            generator_id=chat_data["chat_generator_id"],
        )
        chats.append(chat)

    return chats


# Elastic

ElasticOpType = t.Literal["index", "create", "delete"]
"""Available operations for bulk operations."""

ElasticMapping = {"properties": {"generated": {"type": "nested"}, "messages": {"type": "nested"}}}
"""Default index mapping for chat objects in elastic."""


def chats_to_elastic_data(
    chats: Chat | t.Sequence[Chat],
    index: str,
    *,
    op_type: ElasticOpType = "index",
) -> list[dict[str, t.Any]]:
    """
    Convert chat data to Elasticsearch bulk operation format.

    Args:
        chats: The chat or list of chats to be converted.
        op_type: The operation type for Elasticsearch.

    Returns:
        Formatted bulk operation dict.
    """
    chats = [chats] if isinstance(chats, Chat) else chats

    es_data: list[dict[str, t.Any]] = []
    for chat in chats:
        operation = {"_index": index, "_op_type": op_type, "_id": chat.uuid}
        if op_type != "delete":
            operation["_source"] = chat.model_dump(exclude={"uuid"})
        es_data.append(operation)

    return es_data


async def chats_to_elastic(
    chats: Chat | t.Sequence[Chat],
    index: str,
    client: "elasticsearch.AsyncElasticsearch",
    *,
    op_type: ElasticOpType = "index",
    create_index: bool = True,
    **kwargs: t.Any,
) -> int:
    """
    Convert chat data to Elasticsearch bulk operation format and store it with a client.

    Args:
        chats: The chat or list of chats to be converted and stored.
        index: The name of the Elasticsearch index where the data will be stored.
        client: The AsyncElasticsearch client instance.
        op_type: The operation type for Elasticsearch. Defaults to "create".
        create_index: Whether to create the index if it doesn't exist and update its mapping.
        kwargs: Additional keyword arguments to be passed to the Elasticsearch client.


    Returns:
        The indexed count from the bulk operation
    """
    try:
        import elasticsearch.helpers
    except ImportError as e:
        raise ImportError(
            "Elasticsearch is not available. Please install `elasticsearch` or use `rigging[extra]`.",
        ) from e

    es_data = chats_to_elastic_data(chats, index, op_type=op_type)
    if create_index:
        if (await client.indices.exists(index=index)).meta.status != 200:  # noqa: PLR2004
            await client.indices.create(index=index, mappings=ElasticMapping)
        else:
            await client.indices.put_mapping(index=index, properties=ElasticMapping["properties"])

    results = await elasticsearch.helpers.async_bulk(client, es_data, **kwargs)
    return results[0]  # Return modified count


def elastic_data_to_chats(
    data: "t.Mapping[str, t.Any] | ObjectApiResponse[t.Any]",
) -> list[Chat]:
    """
    Convert the raw elastic results into a list of Chat objects.
    """
    while all(hasattr(data, attr) for attr in ("keys", "__getitem__")) and "hits" in data:
        data = data["hits"]

    objects = t.cast("t.Sequence[t.Mapping[str, t.Any]]", data)
    if not isinstance(objects, t.Sequence):
        raise TypeError(
            f"Expected to find a sequence of objects (optionally under hits), found: {type(data)}",
        )

    chats: list[Chat] = []
    for obj in objects:
        merged = {"uuid": obj["_id"], **obj["_source"]}
        chat = Chat.model_validate(merged)

        # TODO: I don't believe this is safe to deserialize
        # here as we aren't bonded to the underlying rg.Model
        # which was the original object. Skipping for now.
        for msg in chat.all:
            msg.slices = []

        chats.append(chat)

    return chats


async def elastic_to_chats(
    query: t.Mapping[str, t.Any],
    index: str,
    client: "elasticsearch.AsyncElasticsearch",
    *,
    max_results: int | None = None,
    **kwargs: t.Any,
) -> list[Chat]:
    """
    Retrieve chat data from Elasticsearch and convert it to a pandas DataFrame.

    Args:
        query: The Elasticsearch query to be executed.
        index: The name of the Elasticsearch index where the data will be retrieved.
        client: The Elasticsearch client instance.
        max_results: The maximum number of results to retrieve.
        kwargs: Additional keyword arguments to be passed to the Elasticsearch client.

    Returns:
        A pandas DataFrame containing the chat data.
    """
    data = await client.search(index=index, query=query, size=max_results, **kwargs)
    return elastic_data_to_chats(t.cast("dict[str, t.Any]", data))


async def s3_bucket_exists(client: S3Client, bucket: str) -> bool:
    """
    Determine if an S3 bucket exists.

    Args:
        client: The S3 client to use.
        bucket: The bucket to check.

    Returns:
        True if the bucket exists, False otherwise.
    """
    try:
        client.head_bucket(Bucket=bucket)
    except client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

    return True


async def s3_object_exists(client: S3Client, bucket: str, key: str) -> bool:
    """
    Determine if an S3 object exists.

    Args:
        client: The S3 client to use.
        bucket: The bucket to check.
        key: The key to check.

    Returns:
        True if the object exists, False otherwise.
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

    return True
