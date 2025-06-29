import asyncio
import dreadnode as dn
import rigging as rg
from rigging.chat import Chat

async def quality_scorer(chat: Chat) -> float:
    score = 0.0
    text = chat.last.content
    if "climate change" in text.lower():
        score += 0.5
    score += max(0, 1 - (len(text) / 500)) * 0.5
    return score

async def find_best_model_and_output():
    model_ids = [
        "groq/qwen/qwen3-32b",
        "groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        "groq/deepseek-r1-distill-llama-70b"
    ]

    best_outputs = {}

    with dn.run("model-bakeoff") as run:
        for model_id in model_ids:
            print(f"--- Evaluating model: {model_id} ---")

            # Create a pipeline variant for this model
            pipeline = rg.get_generator(model_id).chat(
                "Briefly summarize the key challenges in tackling climate change in three bullet points."
            )

            model_task = pipeline.task(
                label="summary-quality",
                name=model_id,
                log_output=True,  # We want to log the output for comparison
            ).run(scorers=[quality_scorer])

            # Run the task 5 times and get the single best output
            # The .top_n() method handles the map, sort, and selection
            try:
                top_chats = await model_task.top_n(5, 1)
                if top_chats:
                    best_outputs[model_id] = top_chats[0]
                    print(f"Found a top output for {model_id}")
            except Exception as e:
                print(f"Failed to run {model_id}: {e}")

    # Now you can compare the best output from each model
    print("\n--- Best output from each model ---")
    for model_id, chat in best_outputs.items():
        print(f"\nModel: {model_id}\n{chat.last.content}\n")

if __name__ == "__main__":
    dn.configure(server="https://platform.dreadnode.io")
    asyncio.run(find_best_model_and_output())
