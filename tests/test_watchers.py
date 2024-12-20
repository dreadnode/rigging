import json

import pytest

import rigging as rg
from rigging.chat import Chat
from rigging.message import Message


@pytest.fixture
def sample_chats():
    chat1 = Chat(messages=[Message(role="user", content="Hello"), Message(role="assistant", content="Hi there!")])
    chat2 = Chat(
        messages=[Message(role="user", content="How are you?"), Message(role="assistant", content="I'm doing well!")]
    )
    return [chat1, chat2]


@pytest.mark.asyncio
async def test_write_chats_to_jsonl(tmp_path, sample_chats):
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
            for i, message in enumerate(saved_chat["messages"]):
                assert message["role"] == original_chat.messages[i].role
                assert message["content"] == original_chat.messages[i].content


@pytest.mark.asyncio
async def test_write_chats_to_jsonl_append(tmp_path, sample_chats):
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
async def test_write_chats_to_jsonl_replace(tmp_path, sample_chats):
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
        for i, message in enumerate(saved_chat["messages"]):
            assert message["role"] == original_chat.messages[i].role
            assert message["content"] == original_chat.messages[i].content

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
                assert message["content"] == original_chat.messages[j].content


class MockS3Client:
    class exceptions:
        class ClientError(Exception):
            def __init__(self, code: str):
                self.response = {"Error": {"Code": code}}

    class Body:
        def __init__(self, content: str):
            self.content = content

        def read(self):
            return self.content.encode()

    def __init__(self):
        self.buckets = {"test-bucket": {}}

    def head_object(self, Bucket: str, Key: str):
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")
        if Key not in self.buckets[Bucket]:
            raise self.exceptions.ClientError("404")
        return self.buckets[Bucket][Key]

    def get_object(self, Bucket: str, Key: str):
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")
        if Key not in self.buckets[Bucket]:
            raise self.exceptions.ClientError("404")
        return {"Body": MockS3Client.Body(self.buckets[Bucket][Key])}

    def delete_object(self, Bucket: str, Key: str):
        if Bucket not in self.buckets:
            raise self.exceptions.ClientError("404")
        if Key not in self.buckets[Bucket]:
            raise self.exceptions.ClientError("404")
        del self.buckets[Bucket][Key]

    def put_object(self, Bucket: str, Key: str, Body: str):
        self.buckets[Bucket][Key] = Body


@pytest.mark.asyncio
async def test_write_chats_to_s3(sample_chats):
    s3_mock_client = MockS3Client()

    bucket = "test-bucket"
    key = "test/chats.jsonl"

    expected_content = ""
    for chat in sample_chats[:1]:
        expected_content += chat.model_dump_json() + "\n"

    watcher = rg.watchers.write_chats_to_s3(s3_mock_client, bucket, key)

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
    watcher = rg.watchers.write_chats_to_s3(s3_mock_client, bucket, key, replace=True)

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
