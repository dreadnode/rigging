from __future__ import annotations

import asyncio
import sys
import typing as t

from colorama import Fore, Style

from rigging.chat import ChatPipeline
from rigging.generator.base import Generator, get_generator

if t.TYPE_CHECKING:
    from rigging.chat import Chat


async def _animate(*, delay: float = 0.5, chars: list[str] | None = None, color: str = Fore.BLUE) -> None:
    chars = chars or ["   ", ".  ", ".. ", "..."]
    i = 0
    while True:
        print(f"{color}{chars[i]}{Style.RESET_ALL}", end="\r")
        sys.stdout.flush()
        await asyncio.sleep(delay)
        i = (i + 1) % len(chars)


async def interact(
    entrypoint: ChatPipeline | Generator | str, *, reset_callback: t.Callable[[Chat | None], None] | None = None
) -> Chat | None:
    """
    Start an interactive chat session using the given pipeline, generator, or generator id.

    This function allows the user to have a conversation with an assistant by providing input
    and receiving responses. The chat session can be controlled using specific commands.

    Args:
        entrypoint: A ChatPipeline, Generator, or generator id to use for the chat session.
        reset_callback: A callback function to execute when the chat is reset.

    Returns:
        The final Chat object, or None if the chat was interrupted before any generation.
    """

    print(f"\n{Fore.YELLOW}")
    print("Starting interactive chat.")
    print()
    print(" - Type 'exit' to quit or use Ctrl+C.")
    print(" - Type 'reset' or 'restart' to restart the chat.")
    print(" - Type 'again' or 'retry' to re-run the last generation.")
    print(Style.RESET_ALL)

    base_pipeline = (
        entrypoint
        if isinstance(entrypoint, ChatPipeline)
        else entrypoint.chat()
        if isinstance(entrypoint, Generator)
        else get_generator(entrypoint).chat()
    )

    pipeline = base_pipeline.clone()
    chat: Chat | None = None

    while True:
        try:
            user_input = input(f"\n{Fore.GREEN}User: {Style.RESET_ALL}")
            if not user_input:
                continue

            if user_input.lower() == "exit":
                print(f"\n\n{Fore.YELLOW}Exiting chat.{Style.RESET_ALL}")
                break

            if user_input.lower() in ["reset", "restart"]:
                print(f"\n{Fore.YELLOW}--- Reset ---{Style.RESET_ALL}")
                pipeline = base_pipeline.clone()
                if reset_callback:
                    reset_callback(chat)
                continue

            if user_input.lower() in ["again", "retry"]:
                print(f"\n{Fore.YELLOW}--- Retry ---{Style.RESET_ALL}")
                pipeline.chat.messages = pipeline.chat.messages[:-1]
            else:
                pipeline.add(user_input)

            print("")

            animation_task = asyncio.create_task(_animate())
            chat = await pipeline.run()
            animation_task.cancel()

            print(f"\r{Fore.BLUE}Assistant: {Style.RESET_ALL}{chat.last.content}")

            pipeline.add(chat.last)

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}Chat interrupted. Exiting.{Style.RESET_ALL}")
            break
        except Exception as e:
            print(f"\n\n{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
            break

    return chat
