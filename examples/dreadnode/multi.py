import asyncio
import dreadnode as dn
import rigging as rg
from rigging.chat import Chat

dn.configure(server="https://platform.dreadnode.io")

async def creative_agent_workflow() -> None:
    brainstorm_pipeline = rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct").chat(
        "Generate a single, creative blog post title about the future of remote work. Be concise."
    )

    writer_pipeline_template = rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct").chat(
        [
            {"role": "system", "content": "You are an expert author. Your task is to write a compelling 3-paragraph blog post based on the title provided by the user. Do not add any conversational fluff, just write the article."},
            {"role": "user", "content": "Title: {topic}"}
        ]
    )

    async def write_article(topic: str) -> Chat:
        return await writer_pipeline_template.apply(topic=topic).run()

    with dn.run("multi-stage-agent"):
        brainstorm_task = brainstorm_pipeline.task(label="agent-skills").run(name="brainstormer")
        brainstorm_span = await brainstorm_task.run()
        if brainstorm_span.output.failed:
            print(f"Brainstorming pipeline failed gracefully: {brainstorm_span.output.error}")
            return

        topic = brainstorm_span.output.last.content.strip()
        print(f"Generated Topic: {topic}")

        writer_task = dn.task(label="agent-skills", name="writer")(write_article)
        article_span = await writer_task.run(topic=topic)

        if article_span.output.failed:
            print(f"Writing pipeline failed gracefully: {article_span.output.error}")
            return

        print("\n--- Final Article ---")
        print(article_span.output.last.content)

if __name__ == "__main__":
    asyncio.run(creative_agent_workflow())
