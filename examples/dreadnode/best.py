import asyncio
import dreadnode as dn
import rigging as rg
from rigging.chat import Chat

def contains_keyword_scorer(keyword: str):
    async def scorer(chat: Chat) -> float:
        return 1.0 if keyword in chat.last.content.lower() else 0.0
    return scorer

async def find_best_prompt() -> None:
    prompt_candidates = {
        "formal": "You are a professional marketing assistant. Your tone should be formal and persuasive.",
        "casual": "You're a friendly marketing buddy. Be enthusiastic and use casual language.",
        "witty": "You are a witty copywriter. Your goal is to be clever and memorable.",
    }

    base_pipeline = rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct").chat(
        "Write a short tagline for a new brand of sparkling water called 'Crisp'."
    )

    results = {}

    with dn.run("prompt-tuning-experiment") as run:
        for name, system_prompt in prompt_candidates.items():
            print(f"--- Testing prompt: {name} ---")

            prompt_pipeline = base_pipeline.fork(
                rg.Message(role="system", content=system_prompt)
            )

            prompt_task = prompt_pipeline.task(
                label="tagline-generation",
                name=f"prompt-variant-{name}",
                attributes={"system_prompt": system_prompt}
            ).run(scorers=[contains_keyword_scorer("sparkling")])

            spans = await prompt_task.map_run(5)

            all_scores = [s.get_average_metric_value() for s in spans]
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

            results[name] = avg_score
            print(f"Average score for '{name}': {avg_score:.2f}")

    best_prompt = max(results, key=results.get)
    print(f"\n🏆 Best prompt is '{best_prompt}' with a score of {results[best_prompt]:.2f}")

if __name__ == "__main__":
    dn.configure(server="https://platform.dreadnode.io")
    asyncio.run(find_best_prompt())
