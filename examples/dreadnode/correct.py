import rigging as rg
import dreadnode as dn
from rigging.chat import Chat
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    """A model to hold extracted user profile information."""
    name: str = Field(description="The user's full name.")
    email: str = Field(description="The user's email address.")
    city: str = Field(description="The city where the user resides.")

async def pydantic_parse_scorer(chat: Chat) -> float:
    try:
        chat.last.parse(model=UserProfile)
        return 1.0
    except Exception:
        return 0.0

async def extract_structured_data():
    unstructured_text = (
        "Hi there, my name is Jane Doe and I'm a software engineer. "
        "You can reach me at jane.d@email.com. I'm based out of San Francisco."
    )

    extraction_pipeline = (
        rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct")
        .chat(
            f"Extract the user's name, email, and city from the following text and format it as a JSON object. "
            f"Text: '{unstructured_text}'"
        )
        .until_parsed_as(UserProfile)
    )

    with dn.run("structured-extraction-experiment"):
        extraction_task = extraction_pipeline.task(
            label="user-profile-extraction"
        ).run(name="pydantic-extractor", scorers=[pydantic_parse_scorer])

        print("--- Running structured extraction task ---")
        task_span = await extraction_task.run()

        if task_span.failed:
            print("\n--- Task Failed ---")
            print(f"Error: {task_span.error}")
            return

        print("\n--- Final Raw Output ---")
        print(task_span.output.last.content)

        # 5. We can now confidently parse and use the structured data
        final_chat = task_span.output
        try:
            profile = final_chat.last.parse(model=UserProfile)
            print("\n--- Successfully Parsed Pydantic Model ---")
            print(profile.model_dump_json(indent=2))
            print(f"User's name is {profile.name}")
        except Exception as e:
            print(f"\n--- Final Parsing Failed (this shouldn't happen!) ---")
            print(f"Error: {e}")


# Run the structured data extraction
if __name__ == "__main__":
    dn.configure(server="https://platform.dreadnode.io")
    import asyncio
    asyncio.run(extract_structured_data())
