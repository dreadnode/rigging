import typing as t
import rigging as rg


def chat_contains_regex(
    pattern: str, ignore_case: bool = True
) -> t.Callable[[rg.Chat], rg.Chat]:
    """Function that returns an evaluator checking if the output contains a regex pattern."""

    def evaluate(chat: rg.Chat) -> rg.Chat:
        import re

        flags = re.IGNORECASE if ignore_case else 0

        results = []

        for message in chat.messages:
            if message.role == "user":
                match = re.search(pattern, message.content, flags)
                reward = {"value": pattern, "score": float(match is not None)}
                results.append(reward)

        chat.metadata[pattern] = results

        return chat

    return evaluate
