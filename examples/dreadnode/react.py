import rigging as rg
import dreadnode as dn
from rigging.chat import Chat
from typing import Annotated


async def search_engine(
    query: Annotated[str, "A search query to find information."]
) -> str:
    """A search engine to find general information on the web."""
    print(f"TOOL CALLED: search_engine(query='{query}')")
    if "company that makes pixel phone" in query.lower():
        return "Google (part of Alphabet Inc., ticker: GOOGL) makes the Pixel phone."
    return "I couldn't find an answer for that query."

async def get_market_cap(
    ticker: Annotated[str, "The stock ticker symbol of a company, e.g., 'GOOGL'."]
) -> str:
    """Gets the market capitalization of a company."""
    print(f"TOOL CALLED: get_market_cap(ticker='{ticker}')")
    if ticker == "GOOGL":
        return "Alphabet Inc. (GOOGL) has a market cap of approximately $2.1 Trillion."
    return f"Market cap for {ticker} not found."

async def get_latest_news(
    company_name: Annotated[str, "The name of the company to get news for."]
) -> str:
    """Gets the latest news headlines for a specific company."""
    print(f"TOOL CALLED: get_latest_news(company_name='{company_name}')")
    if "google" in company_name.lower() or "alphabet" in company_name.lower():
        return "Latest news for Google: 'Google announces major advancements in AI at their annual developer conference.'"
    return f"No recent news found for {company_name}."


# --- The ReAct Loop Controller ---

async def react_loop_controller(chat: Chat) -> Chat | None:
    """
    This `.then()` callback checks the LLM's last response.
    - If it contains a tool call, it lets the standard tool-handling logic proceed.
    - If it contains the phrase "Final Answer:", the loop terminates.
    - Otherwise, it prompts the LLM to continue its thought process.
    """
    last_message = chat.last.content.strip()

    # The .using() handler already ran. If there were tool calls, they were processed.
    # We only need to decide if the loop should continue.

    if "Final Answer:" in last_message:
        print("✅ Agent has reached a final answer. Terminating loop.")
        return None # Returning None stops the callback chain.

    # If the last message was a tool result, the LLM needs to process it.
    if chat.last.role == "tool":
        print("🔄 Agent processed a tool result. Continuing thought process...")
        # .continue_() re-runs the pipeline with the full history
        return await chat.continue_("Continue with your thought process. If you have enough information, provide the 'Final Answer:'. Otherwise, use another tool.").run()

    print("Agent hasn't finished. This should not be hit often if prompts are good.")
    return None


async def run_react_agent():
    # A detailed system prompt is crucial for ReAct agents
    system_prompt = """
    You are a helpful research assistant. To answer the user's question, you must use a strict "Thought, Action, Observation" loop.
    1.  **Thought:** Briefly reason about what you need to do next.
    2.  **Action:** Call ONE of the available tools (`search_engine`, `get_market_cap`, `get_latest_news`).
    3.  **Observation:** After the tool returns its result, you will see it.
    4.  Repeat this process until you have enough information.
    5.  When you have the final answer, prefix it with "Final Answer:".
    """

    # 1. Define the agent pipeline with all its tools and the loop controller
    agent_pipeline = (
        rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct")
        .chat([
            rg.Message("system", system_prompt),
            rg.Message("user", "What is the market cap of the company that makes the Pixel phone, and what is their latest news?")
        ])
        .using(search_engine, get_market_cap, get_latest_news)
        .then(react_loop_controller)
    )

    with dn.run("react-agent-experiment"):
        # 2. Create the Task from the agent pipeline
        agent_task = agent_pipeline.task(
            label="complex-research-queries"
        ).run(name="multi-tool-react-agent")

        # 3. Execute the agent task
        print("--- Running ReAct Agent ---")
        task_span = await agent_task.run()

        print("\n\n--- Agent's Final Output ---")
        final_answer = task_span.output.last.content.replace("Final Answer:", "").strip()
        print(final_answer)

        print("\n--- Full Conversation ---")
        print(task_span.output.conversation)


if __name__ == "__main__":
    dn.configure(server="https://platform.dreadnode.io")

    import asyncio
    asyncio.run(run_react_agent())
