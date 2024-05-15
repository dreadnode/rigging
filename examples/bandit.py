import abc
import asyncio
import pathlib
import random
import re
import typing as t
from dataclasses import dataclass, field

import asyncssh
import click
import requests  # type: ignore
from loguru import logger
from pydantic import StringConstraints

import rigging as rg
from rigging import logging

# Constants

SSH_HOST = "bandit.labs.overthewire.org"
SSH_PORT = 2220

MAX_PINS = 10
MAX_HISTORY = 5

SYSTEM_PROMPT = """\
You are a senior security expert with years of linux and CTF experience. You are solving a CTF which involves
locating a password on a server by executing commands via SSH.
"""

# Helpers


def get_bandit_level_description(level: int) -> str:
    search = r"Level Goal</h2>(.+)<h2"
    response = requests.get(f"https://overthewire.org/wargames/bandit/bandit{level}.html")
    response.raise_for_status()
    goal: str = re.findall(search, response.text, re.DOTALL)[0]
    goal = goal.replace("<p>", "").replace("</p>", "").strip()
    return re.sub("<.*?>", "", goal)


async def connect_ssh(level: int, password: str) -> asyncssh.SSHClientConnection | None:
    username = f"bandit{level}"

    try:
        conn = await asyncssh.connect(SSH_HOST, SSH_PORT, username=username, password=password, known_hosts=None)
        logger.success(f"Connected to {SSH_HOST} on port {SSH_PORT} as {username}")
        return conn
    except Exception as e:
        logger.error(f"Failed to authenticate as {username} with password {password}")
        logger.error(str(e))
        return None


async def execute_ssh(
    conn: asyncssh.SSHClientConnection, command: str, *, max_output_len: int = 5_000, timeout: int = 10
) -> str:
    logger.debug(f"Executing:\n{command}")

    async with conn.create_process("/bin/bash") as process:  # type: ignore
        process.stdin.write(command + "\n" + "exit" + "\n")
        try:
            stdout_output, stderr_output = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.terminate()
            stdout_output = ""
            stderr_output = "[command timed out]"

    output = f"{stdout_output}\n{stderr_output}".strip()

    if not output:
        output = "[command finished]"

    if len(output) > max_output_len:
        output = output[:max_output_len] + "\n[output truncated]"

    logger.debug(f"Output:\n{output}")

    return output


# Models

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]


class Action(rg.Model, abc.ABC):
    @abc.abstractmethod
    async def run(self, state: "State") -> str:
        ...


class UpdateGoal(Action):
    goal: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return UpdateGoal(goal="My new goal").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"[{state.id}] Updating goal to '{self.goal}'")
        state.goals.append(self.goal)
        return "Goal updated."


class SaveMemory(Action):
    key: str_strip = rg.attr()
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return SaveMemory(key="my-note", content="Lots of custom data\nKeep this for later.").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"[{state.id}] Storing '{self.key}':\n{self.content}")
        state.memories[self.key] = self.content
        return f"Stored '{self.key}'."


class RecallMemory(Action):
    key: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return RecallMemory(key="last-thoughts").to_pretty_xml()

    async def run(self, state: "State") -> str:
        value = state.memories.get(self.key, "Not found.")
        logger.success(f"[{state.id}] Recalling '{self.key}'\n{value}")
        return value


class DeleteMemory(Action):
    key: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return DeleteMemory(key="my-note").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"[{state.id}] Forgetting '{self.key}'")
        state.memories.pop(self.key, None)
        return f"Forgot '{self.key}'."


class PinToTop(Action):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return PinToTop(content="This is the auth token: 1234").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"[{state.id}] Pinning '{self.content}'")
        state.pins.append(self.content)
        state.pins = state.pins[:MAX_PINS]
        return "Pinned."


class TryCommand(Action):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return TryCommand(content="whoami | grep abc").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.info(f"[{state.id}] Trying command:\n{self.content}")
        assert state.client is not None, "No SSH connection available"
        return await execute_ssh(state.client, self.content)


class SubmitPassword(Action):
    password: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return SubmitPassword(password="[long_pw_string]").to_pretty_xml()

    async def run(self, state: "State") -> str:
        if re.search(r"[a-zA-Z0-9]{32}", self.password) is None:
            return "Invalid password format."

        next_level = state.level + 1
        client = await connect_ssh(next_level, self.password)
        if client is None:
            return "Failed to connect. Invalid password."

        logger.success(f"[{state.id}] Got password for level {next_level}: {self.password}")
        state.finish(self.password)

        return f"Success! You are now on level {next_level}."


Actions = UpdateGoal | SaveMemory | RecallMemory | DeleteMemory | PinToTop | TryCommand | SubmitPassword
ActionsList: list[type[Actions]] = [
    UpdateGoal,
    SaveMemory,
    RecallMemory,
    DeleteMemory,
    PinToTop,
    TryCommand,
    SubmitPassword,
]


@dataclass
class State:
    # Required
    id: int
    max_actions: int
    base_chat: rg.PendingChat

    # Progress
    result: str | None = ""

    # CTF
    client: asyncssh.SSHClientConnection | None = None
    level: int = 1
    level_details: str = ""

    # Core
    goals: list[str] = field(default_factory=list)
    next_actions: list[Actions] = field(default_factory=list)

    # Context
    pins: list[str] = field(default_factory=list)
    memories: dict[str, str] = field(default_factory=dict)
    history: list[tuple[Actions, str]] = field(default_factory=list)

    def finish(self, password: str) -> None:
        self.result = password

    async def prep(self, level: int, password: str) -> None:
        self.client = await connect_ssh(level, password)
        self.level = level
        self.level_details = get_bandit_level_description(level + 1)
        self.goals.append(f"Find and submit the password to the next level ({self.level + 1})")

    async def step(self) -> None:
        logger.debug(f"Processing {len(self.next_actions)} action(s)")
        for action in self.next_actions:
            self.history.append((action, await action.run(self)))
        self.next_actions.clear()

    def get_prompt(self, max_history: int = MAX_HISTORY) -> str:
        memories = "\n".join(self.memories.keys())
        previous_goals = "\n".join(self.goals[:-1] if len(self.goals) > 1 else [])
        current_goal = self.goals[-1]
        history = "\n---\n".join([h[0].to_pretty_xml() + "\n" + h[1] for h in self.history[-max_history:]])
        pinned = "\n".join(self.pins)
        return f"""\
# Context

<current-level>
{self.level}
</current-level>

<current-level-details>
{self.level_details}
</current-level-details>

<memories>
{memories or 'No memories yet.'}
</memories>

<last-{max_history}-actions>
{history or 'No actions taken yet'}
</last-{max_history}-actions>

<pinned>
{pinned or 'No pinned context yet.'}
</pinned>

# Goals

<previous-goals>
{previous_goals or 'No previous goals'}
</previous-goals>

<current-goal>
{current_goal}
</current-goal>

# Actions

You can take any 1 of the following actions in your response. Use the formats below.

## Memory

You can use the memory actions to store and retrieve long term information as you work. Use memories often to keep track of important information like your planning, analysis, important web responses, etc.

To store a memory:
{SaveMemory.xml_example()}

To retrieve a memory:
{RecallMemory.xml_example()}

To forget a memory:
{DeleteMemory.xml_example()}

## Pinned Context

In addition to long term memories, you can pin important short term information to be displayed before every future action (maximum of {MAX_PINS}). Use this to keep data like tokens, passwords, usernames, endpoints, critical notes, etc.

To pin important information:
{PinToTop.xml_example()}

## Goal

When you believe you've accomplished your current goal:
{UpdateGoal.xml_example()}

## Commands

To execute a command on the remote host via SSH:
{TryCommand.xml_example()}

## Password

When you have the password to the next level, provide it so the system can authenticate you to the next level:
{SubmitPassword.xml_example()}

---

# Guidance

- Do not use placeholder values in any of your actions.
- Recall your memories to help you make decisions.
- Experiment frequently and be adjust your approach to make progress.
- All actions are useful, be organized and methodical in your approach.
- Always assume you start in a new /bin/bash shell in the user home directory
- Add verbosity and failure information to your commands to assist with troubleshooting
- Prefer using full paths to files and directories
- Be creative with your solutions, but keep them simple
- Avoid repeating commands that have already been executed and failed
- Read output carefully to understand what errors occured and why
- Use the /tmp directory for any file write operations
- You are executing the commands on the remote host, not locally
- Passwords look like long base64 strings, watch for them

Output a new action from the list above in your response. Prior action results are displayed above.
"""


# CLI  + Core


async def agent_loop(state: State) -> State:
    async def parse_actions(chat: rg.Chat) -> rg.Chat | None:
        parsed: list[Actions] = []
        for action_cls in ActionsList:
            action = chat.last.try_parse(action_cls)
            if action is not None:
                parsed.append(action)  # type: ignore

        if not parsed:
            logger.warning("Model didn't provide any valid actions")
            return None

        parsed = t.cast(list[Actions], [p.model for p in chat.last.parts])
        if len(parsed) > state.max_actions:
            logger.warning("Model provided multiple actions, taking just the first")

        state.next_actions = parsed[: state.max_actions]
        return None

    while not state.result:
        await state.base_chat.fork(state.get_prompt()).then(parse_actions).arun()
        await state.step()

    return state


async def main(
    level: int, password: str, generator_id: str, max_iterations: int, parallel_agents: int, max_actions: int
) -> None:
    logger.success(f"Starting Bandit with {parallel_agents} agents")

    # Prepare our objects

    generator = rg.get_generator(generator_id)
    base_chat = generator.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}],
    ).with_(max_tokens=4096)

    for i in range(max_iterations):
        logger.success(f"Starting level {level}")

        states: list[State] = [
            State(id=i, max_actions=max_actions, base_chat=base_chat.with_(temperature=random.uniform(0.25, 1)))
            for i in range(parallel_agents)
        ]
        for state in states:
            await state.prep(level, password)

        loops = [asyncio.create_task(agent_loop(state)) for state in states]
        _, pending = await asyncio.wait(loops, return_when=asyncio.FIRST_COMPLETED)

        for task in pending:
            task.cancel()

        finished_state = next(s for s in states if s.result)
        level = finished_state.level + 1
        password = finished_state.result or ""

    logger.success("Finished Bandit.")


@click.command()
@click.option("-l", "--level", type=int, default=1, help="Starting level (1-34)")
@click.option("-p", "--password", type=str, default="NH2SXQwcBdpmTEzi3bvBHMM9H66vVXjL", help="Starting password")
@click.option(
    "-g",
    "--generator-id",
    type=str,
    default="anthropic/claude-3-sonnet-20240229",
    required=True,
    help="Rigging generator identifier (gpt-4, mistral/mistral-medium, etc.)",
)
@click.option(
    "-i",
    "--max-iterations",
    type=int,
    default=100,
    help="Maximum number of iterations",
)
@click.option(
    "-p",
    "--parallel-agents",
    type=int,
    default=3,
    help="Number of parallel agents",
)
@click.option(
    "-m",
    "--max-actions",
    type=int,
    default=3,
    help="Maximum number of actions allowed per generation round",
)
@click.option(
    "--log-level",
    type=click.Choice(logging.LogLevelList),
    default="info",
)
@click.option("--log-file", type=click.Path(path_type=pathlib.Path), default="bandit.log")
@click.option(
    "--log-file-level",
    type=click.Choice(logging.LogLevelList),
    default="trace",
)
def cli(
    level: int,
    password: str,
    generator_id: str,
    max_iterations: int,
    parallel_agents: int,
    max_actions: int,
    log_level: logging.LogLevelLiteral,
    log_file: pathlib.Path,
    log_file_level: logging.LogLevelLiteral,
) -> None:
    """
    Rigging example for agentic exploitation of OverTheWire's Bandit wargame.
    """

    logging.configure_logging(log_level, log_file, log_file_level)
    asyncio.run(main(level, password, generator_id, max_iterations, parallel_agents, max_actions))


if __name__ == "__main__":
    cli()
