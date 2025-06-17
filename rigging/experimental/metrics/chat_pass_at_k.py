import math
import typing as t

import rigging as rg


async def chat_pass_at_k(
    values: list[rg.Model],
    message: t.Literal["system", "assistant", "tool"],
    k: list = [1, 2, 5, 10],
) -> t.Callable[[rg.Chat], rg.Chat]:
    """
    Calculate pass@k

    Args:
        n: total number of samples
        c: number of correct samples
        k: number of samples to evaluate

    Returns:
        pass@k probability
    """

    async def evaluate(chat: rg.Chat) -> rg.Chat:
        num_samples = len(values)
        num_correct_samples = 0
        num_samples_to_evaluate = 0
        if num_correct_samples >= num_samples_to_evaluate:
            return 1.0
        if num_correct_samples == 0:
            return 0.0
        if num_samples_to_evaluate > num_samples:
            return None  # Invalid case

        result = 1.0 - math.comb(
            num_samples - num_correct_samples, num_samples_to_evaluate
        ) / math.comb(num_samples, num_samples_to_evaluate)

        chat.metadata["pass_at_k"] = result

        return chat

    return evaluate
