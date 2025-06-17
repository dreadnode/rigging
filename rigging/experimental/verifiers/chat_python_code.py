import typing as t

import rigging as rg


async def chat_python_code(
    output: str,
) -> t.Callable[[rg.Chat], bool]:
    """
    Evaluator that uses python code to evaluate the result.
    """

    async def evaluate(chat: rg.Chat) -> rg.Chat:
        print("Evaluating chat with python code...")
        return chat

    return evaluate
