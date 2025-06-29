import asyncio
from typing import Annotated, cast

import dreadnode as dn
import rigging as rg
from rigging.chat import Chat
from rigging import Tool

# --- Mock Tools (same as before) ---

async def search_engine(query: str) -> str:
    """A search engine to find general information on the web."""
    print(f"TOOL CALLED: search_engine(query='{query}')")
    if "company that makes pixel phone" in query.lower():
        return "Google (part of Alphabet Inc., ticker: GOOGL) makes the Pixel phone."
    return "I couldn't find an answer for that query."

async def get_market_cap(ticker: str) -> str:
    """Gets the market capitalization of a company."""
    print(f"TOOL CALLED: get_market_cap(ticker='{ticker}')")
    if ticker == "GOOGL":
        return "Alphabet Inc. (GOOGL) has a market cap of approximately $2.1 Trillion."
    return f"Market cap for {ticker} not found."

async def get_latest_news(company_name: str) -> str:
    """Gets the latest news headlines for a specific company."""
    print(f"TOOL CALLED: get_latest_news(company_name='{company_name}')")
    if "google" in company_name.lower() or "alphabet" in company_name.lower():
        return "Latest news for Google: 'Google announces major advancements in AI at their annual developer conference.'"
    return f"No recent news found for {company_name}."

# Store our tools in a dictionary for easy lookup
available_tools = {
    "search_engine": Tool.from_callable(search_engine),
    "get_market_cap": Tool.from_callable(get_market_cap),
    "get_latest_news": Tool.from_callable(get_latest_news),
}


# --- The Corrected ReAct Step Executor ---

async def react_step_executor(chat: Chat) -> Chat | None:
    """
    This `.then()` callback executes ONE step of the ReAct loop.
    It handles tool calls and then recursively calls the pipeline
    to continue the loop until a final answer is found.
    """
    print(f"\n--- ReAct Step ---")
    last_message = chat.last

    # Condition 1: Check for Final Answer (Termination)
    if "Final Answer:" in last_message.content:
        print("✅ Agent has reached a final answer. Terminating loop.")
        return None # Returning None ends the recursive chain.

    # Condition 2: Check for Tool Calls (Action)
    if not last_message.tool_calls:
        print("⚠️ Agent did not call a tool or provide a final answer. Stopping.")
        return None # Stop if the LLM gets stuck.

    # --- Handle Tool Calls ---
    # We manually process the tool calls from the last message
    tool_messages = []
    for tool_call in last_message.tool_calls:
        if tool := available_tools.get(tool_call.name):
            message, _ = await tool.handle_tool_call(tool_call)
            tool_messages.append(message)
        else:
            tool_messages.append(rg.Message("tool", f"Error: Tool '{tool_call.name}' not found.", tool_call_id=tool_call.id))

    # --- Continue the Loop (Recursion) ---
    # Create the next step in the pipeline by appending the tool results
    # and re-running the same pipeline logic.
    print("🔄 Agent used a tool. Continuing the thought process...")

    # We use .continue_() to restart the pipeline with the full history,
    # including the tool results we just generated.
    # The pipeline it returns already has our `react_step_executor` callback attached.
    next_step_chat = await chat.continue_(tool_messages).run()

    # By returning the result of the next step, we replace the current
    # chat with the fully completed one from the end of the recursive chain.
    return next_step_chat


async def run_final_react_agent():
    system_prompt = """
    You are a helpful research assistant. To answer the user's question, you must use a strict "Thought, Action, Observation" loop.
    1.  **Thought:** Briefly reason about what you need to do next.
    2.  **Action:** Call ONE of the available tools.
    3.  **Observation:** After the tool returns its result, you will see it.
    4.  Repeat this process until you have enough information.
    5.  When you have the final answer, prefix it with "Final Answer:".
    """

    # 1. Define the base pipeline. We will NOT use `.using()`.
    base_pipeline = rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct").chat([
        rg.Message("system", system_prompt),
        rg.Message("user", "What is the market cap of the company that makes the Pixel phone, and what is their latest news?")
    ])

    # Manually add tool definitions to the params for the LLM to see
    base_pipeline.params = rg.GenerateParams(
        tools=[tool.api_definition for tool in available_tools.values()]
    )

    # 2. Add our recursive executor to the pipeline
    agent_pipeline = base_pipeline.then(react_step_executor)

    with dn.run("final-react-agent-experiment"):
        # 3. Create the Task. The definition is identical.
        agent_task = agent_pipeline.task(
            label="final-research-queries"
        ).run(name="recursive-react-agent")

        # 4. Execute the agent task
        print("--- Running Final ReAct Agent ---")
        task_span = await agent_task.run()

        print("\n\n--- Agent's Final Output ---")
        final_answer = task_span.output.last.content.replace("Final Answer:", "").strip()
        print(final_answer)

        print("\n--- Full Conversation ---")
        print(task_span.output.conversation)


if __name__ == "__main__":
    dn.configure(server="https://platform.dreadnode.io")

    # Run the final ReAct agent
    asyncio.run(run_final_react_agent())
