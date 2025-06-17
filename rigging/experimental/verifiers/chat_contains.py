import rigging as rg


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
