import asyncio
import os

import litellm
import logfire

import rigging as rg

# Constants
SYSTEM_PROMPT = "Enumerate all open TCP ports on 127.0.0.1 and provide a vulnerability report"

# LOGFIRE_PROJECT = "rigging-demos"
logfire.configure()
os.environ.setdefault("LOGFIRE_TOKEN", "")  # (1)!
litellm.callbacks = ["logfire"]

rg.logging.configure_logging("debug")


async def main():
    tools = rg.integrations.robopages("http://localhost:8000")
    chat = await rg.get_generator("gpt-4").chat(SYSTEM_PROMPT).using(*tools).run()
    print(chat.conversation)


if __name__ == "__main__":
    asyncio.run(main())
