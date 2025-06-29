import random
from typing import Annotated
import rigging as rg
import dreadnode as dn

# Define a simple async tool the LLM can use
async def get_stock_price(
    symbol: Annotated[str, "The stock ticker symbol, e.g., 'ACME'"]
) -> float:
    """Gets the current price of a stock."""
    price = 100.0 + random.uniform(-5, 5)
    return round(price, 2)


async def stock_price_agent():
    agent_pipeline = (
        rg.get_generator("groq/meta-llama/llama-4-maverick-17b-128e-instruct")
        .chat("What is the current stock price of the COKE company?")
        .using(get_stock_price)
    )

    with dn.run("tool-using-agent-experiment"):
        agent_task = agent_pipeline.task(
            label="stock-queries"
        ).run(name="stock-price-agent")

        print("--- Running agent task ---")
        chat = await agent_task.run()

        print("\n--- Final Answer ---")
        print(chat.output.last.content)


if __name__ == "__main__":
    dn.configure(server="https://platform.dreadnode.io")

    import asyncio
    asyncio.run(stock_price_agent())
