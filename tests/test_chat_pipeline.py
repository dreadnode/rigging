import typing as t

import pytest

from rigging import Chat, Message
from rigging.chat import PipelineStepContextManager
from rigging.error import MaxDepthError
from rigging.generator import GenerateParams
from rigging.model import YesNoAnswer

from .generators import (
    CallbackGenerator,
    FailingGenerator,
    FixedGenerator,
)

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001


@pytest.mark.asyncio()
async def test_basic_chat_pipeline() -> None:
    generator = FixedGenerator(
        model="fixed",
        text="This is a fixed response",
        params=GenerateParams(),
    )
    chat = await generator.chat([{"role": "user", "content": "Hello"}]).run()

    assert len(chat) == 2
    assert chat.messages[0].role == "user"
    assert chat.messages[0].content == "Hello"
    assert chat.generated[0].role == "assistant"
    assert chat.generated[0].content == "This is a fixed response"
    assert chat.stop_reason == "stop"
    assert not chat.failed


@pytest.mark.asyncio()
async def test_chat_until_parsed_as_basic() -> None:
    generator = FixedGenerator(
        model="fixed",
        text="<yes-no-answer>yes</yes-no-answer>",
        params=GenerateParams(),
    )

    chat = (
        await generator.chat([{"role": "user", "content": "Is the sky blue?"}])
        .until_parsed_as(YesNoAnswer)
        .run()
    )

    assert len(chat) == 2
    assert chat.last.content == "<yes-no-answer>yes</yes-no-answer>"
    assert chat.last.try_parse(YesNoAnswer) is not None
    assert chat.last.parse(YesNoAnswer).boolean is True
    assert not chat.failed


@pytest.mark.asyncio()
async def test_max_depth_limit() -> None:
    max_depth = 3
    call_count = 0

    async def recursive_callback(chat: Chat) -> PipelineStepContextManager:
        nonlocal call_count
        call_count += 1
        return chat.restart().step()

    generator = FixedGenerator(model="fixed", text="test response", params=GenerateParams())

    chat = (
        await generator.chat("Hello")
        .then(recursive_callback, max_depth=max_depth)
        .run(on_failed="include")
    )

    assert chat.failed
    assert isinstance(chat.error, MaxDepthError)
    assert call_count == max_depth + 1  # One initial call + max_depth recursive calls


@pytest.mark.asyncio()
async def test_pipeline_steps() -> None:
    should_continue = True

    async def callback(chat: Chat) -> PipelineStepContextManager | None:
        nonlocal should_continue
        if not should_continue:
            return None
        should_continue = False
        return chat.restart().step()

    generator = FixedGenerator(model="fixed", text="test response", params=GenerateParams())

    async with generator.chat("Hello").step() as steps:
        all_steps = [step async for step in steps]

    assert len(all_steps) == 2
    assert all_steps[0].state == "generated"
    assert all_steps[1].state == "final"

    async with generator.chat("Hello").then(callback).step() as steps:
        all_steps = [step async for step in steps]

    assert len(all_steps) == 6
    assert all_steps[0].state == "generated"
    assert all_steps[1].state == "generated"
    assert all_steps[2].state == "callback"
    assert all_steps[3].state == "final"
    assert all_steps[4].state == "callback"
    assert all_steps[5].state == "final"
    assert len({id(step.pipeline) for step in all_steps}) == 2
    assert all_steps[0].pipeline is not all_steps[1].pipeline
    assert all_steps[0].chats[0].messages[0].content == "Hello"
    assert all_steps[0].chats[0].last.content == "test response"


@pytest.mark.asyncio()
async def test_chat_pipeline_error_handling() -> None:
    generator = FailingGenerator(model="failing", params=GenerateParams())

    # Test "raise" mode (default)
    with pytest.raises(RuntimeError):
        await generator.chat("test").run()

    # Should still raise if the error is not in the default list
    with pytest.raises(RuntimeError):
        await generator.chat("test").run(on_failed="include")

    # Now add it to the catch list
    chat_catch = await generator.chat("test").catch(RuntimeError).run(on_failed="include")
    assert chat_catch.failed
    assert isinstance(chat_catch.error, RuntimeError)

    # Check for default error handling
    generator._exception = MaxDepthError(0, None, "")  # type: ignore [arg-type]
    chat_catch = await generator.chat("test").run(on_failed="include")
    assert chat_catch.failed
    assert isinstance(chat_catch.error, MaxDepthError)

    # Skip should fail for single generations
    with pytest.raises(ValueError):
        await generator.chat("test").run(on_failed="skip")

    # No chats if we skip them all
    chat_skip = await generator.chat("test").run_many(5, on_failed="skip")
    assert len(chat_skip) == 0


@pytest.mark.asyncio()
async def test_parsing_with_recovery() -> None:
    response_sequence = [
        "Invalid response",  # First response that fails to parse
        "Still invalid",  # Second response that fails to parse
        "<yes-no-answer>yes</yes-no-answer>",  # Finally a valid response
    ]
    current_response_index = 0

    def get_next_response(self: CallbackGenerator, messages: t.Sequence[Message]) -> str:
        nonlocal current_response_index
        response = response_sequence[current_response_index]
        current_response_index = min(current_response_index + 1, len(response_sequence) - 1)
        return response

    generator = CallbackGenerator(model="callback", params=GenerateParams())
    generator.message_callback = get_next_response

    chat = await generator.chat("test").until_parsed_as(YesNoAnswer).run()

    # Should have gone through recovery and eventually succeeded
    assert not chat.failed
    assert chat.last.try_parse(YesNoAnswer) is not None
    assert chat.last.parse(YesNoAnswer).boolean is True
    assert len(chat.all) > 2  # Should have more than just the initial Q&A due to recovery attempts


@pytest.mark.asyncio()
async def test_map_callback() -> None:
    generator = FixedGenerator(model="fixed", text="Response 1", params=GenerateParams())

    async def double_chats(chats: list[Chat]) -> list[Chat]:
        # Create a copy of each chat with a modified response
        new_chats = []
        for chat in chats:
            new_chat = chat.clone()
            new_chat.generated[0].content = "Modified: " + new_chat.generated[0].content
            new_chats.append(new_chat)
        return chats + new_chats

    chats = (
        await generator.chat([{"role": "user", "content": "Hello"}]).map(double_chats).run_many(1)
    )

    assert len(chats) == 2
    assert chats[0].last.content == "Response 1"
    assert chats[1].last.content == "Modified: Response 1"


@pytest.mark.asyncio()
async def test_watch_callback() -> None:
    generator = FixedGenerator(model="fixed", text="Response", params=GenerateParams())

    watch_calls = []

    async def watch_function(chats: list[Chat]) -> None:
        watch_calls.append(len(chats))

    await generator.chat([{"role": "user", "content": "Hello"}]).watch(watch_function).run()

    # Watch should be called at least once
    assert len(watch_calls) >= 1
    assert all(calls >= 1 for calls in watch_calls)
