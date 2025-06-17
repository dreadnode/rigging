import typing as t
import rigging as rg


def chat_llm_judge(
    # generator: rg.Generator,
    judge_task: str,
) -> t.Callable[[rg.Chat], bool]:
    """
    Evaluator that uses a generator to judge messages in a chat.
    """

    def evaluate(chat: rg.Chat) -> rg.Chat:
        print(f"Evaluating chat with judge task: {judge_task}")
        return chat

    return evaluate
