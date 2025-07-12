import re

from rigging.chat import Chat, ThenChatCallback


def chat_contains(
    pattern: str | re.Pattern[str], *, case_sensitive: bool = False, regex: bool = False
) -> ThenChatCallback:
    """Function that returns an evaluator checking if the output contains a regex pattern."""

    if isinstance(pattern, str):
        if not regex:
            pattern = re.escape(pattern)
        pattern = re.compile(pattern, flags=re.IGNORECASE if not case_sensitive else 0)

    def _chat_contains(chat: Chat) -> Chat:
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

    return _chat_contains


def chat_contains(value: str | rg.Model, ignore_case: bool = False):
    """Function that returns an evaluator checking if the output contains a given value."""

    async def evaluate(chat: rg.Chat) -> rg.Chat:
        results = []
        for message in chat.messages:
            reward = {"value": value, "score": ""}
            if message.role == "user":
                if not ignore_case:
                    result = float(value in message.content)
                    reward["score"] = result
                else:
                    result = float(value.lower() in message.content.lower())
                    reward["score"] = result

                results.append(reward)

        chat.metadata[value] = results

        return chat

    return evaluate
