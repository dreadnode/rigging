import json
import typing as t
from pathlib import Path

import pytest

import rigging as rg
from rigging.chat import Chat
from rigging.message import Message

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001, FBT001, FBT002, N803


@pytest.fixture
def sample_chats() -> list[Chat]:
    chat1 = Chat(
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
    )
    chat2 = Chat(
        messages=[
            Message(role="user", content="How are you?"),
            Message(role="assistant", content="I'm doing well!"),
        ],
    )
    return [chat1, chat2]


@pytest.mark.asyncio
async def test_write_chats_to_jsonl(tmp_path: Path, sample_chats: list[Chat]) -> None:
    output_file = tmp_path / "chats.jsonl"
    watcher = rg.watchers.write_chats_to_jsonl(output_file)

    await watcher(sample_chats)

    assert output_file.exists()

    # read and verify contents
    with output_file.open() as f:
        lines = f.readlines()
        assert len(lines) == 2

        # verify each chat was written correctly
        for i, line in enumerate(lines):
            saved_chat = json.loads(line)
            original_chat = sample_chats[i]
            for k, message in enumerate(saved_chat["messages"]):
                assert message["role"] == original_chat.messages[k].role
                assert message["content"] == [
                    {"text": original_chat.messages[k].content, "type": "text"},
                ]


@pytest.mark.asyncio
async def test_write_chats_to_jsonl_append(tmp_path: Path, sample_chats: list[Chat]) -> None:
    output_file = tmp_path / "chats.jsonl"
    watcher = rg.watchers.write_chats_to_jsonl(output_file)

    # write first batch
    await watcher(sample_chats[:1])

    # write second batch
    await watcher(sample_chats[1:])

    with output_file.open() as f:
        lines = f.readlines()
        assert len(lines) == 2


@pytest.mark.asyncio
async def test_write_chats_to_jsonl_replace(tmp_path: Path, sample_chats: list[Chat]) -> None:
    output_file = tmp_path / "chats.jsonl"

    # write initial content
    watcher1 = rg.watchers.write_chats_to_jsonl(output_file)
    await watcher1(sample_chats)

    # create new watcher with replace=True
    watcher2 = rg.watchers.write_chats_to_jsonl(output_file, replace=True)

    # write only one chat - should replace previous content
    await watcher2(sample_chats[:1])

    with output_file.open() as f:
        lines = f.readlines()
        assert len(lines) == 1
        saved_chat = json.loads(lines[0])
        original_chat = sample_chats[0]
        for k, message in enumerate(saved_chat["messages"]):
            assert message["role"] == original_chat.messages[k].role
            assert message["content"] == [
                {"text": original_chat.messages[k].content, "type": "text"},
            ]

    # write another chat - should append since already replaced once
    await watcher2(sample_chats[1:2])

    with output_file.open() as f:
        lines = f.readlines()
        assert len(lines) == 2

        # verify both chats were written correctly
        for i, line in enumerate(lines):
            saved_chat = json.loads(line)
            original_chat = sample_chats[i]
            for j, message in enumerate(saved_chat["messages"]):
                assert message["role"] == original_chat.messages[j].role
                assert message["content"] == [
                    {"text": original_chat.messages[j].content, "type": "text"},
                ]


class MockS3Client:
    class exceptions:  # noqa: N801
        class ClientError(Exception):
            def __init__(self, code: str):
                self.response = {"Error": {"Code": code}}

    class Body:
        def __init__(self, content: str):
            self.content = content

        def read(self) -> bytes:
            return self.content.encode()

    def __init__(self) -> None:
        self.buckets: dict[str, t.Any] = {"test-bucket": {}}

    def head_object(self, Bucket: str, Key: str) -> t.Any:
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")
        if Key not in self.buckets[Bucket]:
            raise self.exceptions.ClientError("404")
        return self.buckets[Bucket][Key]

    def get_object(self, Bucket: str, Key: str) -> t.Any:
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")
        if Key not in self.buckets[Bucket]:
            raise self.exceptions.ClientError("404")
        return {"Body": MockS3Client.Body(self.buckets[Bucket][Key])}

    def delete_object(self, Bucket: str, Key: str) -> None:
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")
        if Key not in self.buckets[Bucket]:
            raise self.exceptions.ClientError("404")
        del self.buckets[Bucket][Key]

    def put_object(self, Bucket: str, Key: str, Body: str) -> None:
        self.buckets[Bucket][Key] = Body


@pytest.mark.asyncio
async def test_write_chats_to_s3(sample_chats: list[Chat]) -> None:
    s3_mock_client = MockS3Client()

    bucket = "test-bucket"
    key = "test/chats.jsonl"

    expected_content = ""
    for chat in sample_chats[:1]:
        expected_content += chat.model_dump_json() + "\n"

    watcher = rg.watchers.write_chats_to_s3(s3_mock_client, bucket, key)  # type: ignore [arg-type]

    # write first batch
    await watcher(sample_chats[:1])

    got = s3_mock_client.get_object(Bucket=bucket, Key=key)
    assert got["Body"].read() == expected_content.encode()

    # write second batch
    await watcher(sample_chats[1:])

    expected_content = ""
    for chat in sample_chats:
        expected_content += chat.model_dump_json() + "\n"

    got = s3_mock_client.get_object(Bucket=bucket, Key=key)
    assert got["Body"].read() == expected_content.encode()

    # create a new watcher with replace=True
    watcher = rg.watchers.write_chats_to_s3(s3_mock_client, bucket, key, replace=True)  # type: ignore [arg-type]

    # write a single chat
    await watcher(sample_chats[:1])

    expected_content = ""
    for chat in sample_chats[:1]:
        expected_content += chat.model_dump_json() + "\n"

    # verify it's been replaced
    got = s3_mock_client.get_object(Bucket=bucket, Key=key)
    assert got["Body"].read() == expected_content.encode()

    # write second batch
    await watcher(sample_chats[1:])

    expected_content = ""
    for chat in sample_chats:
        expected_content += chat.model_dump_json() + "\n"

    # verify replace happens only once
    got = s3_mock_client.get_object(Bucket=bucket, Key=key)
    assert got["Body"].read() == expected_content.encode()
