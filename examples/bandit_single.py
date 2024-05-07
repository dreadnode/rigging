import abc
import asyncio
import pathlib
import re
import socket
import time
import typing as t
from dataclasses import dataclass, field

import _shared
import click
import paramiko  # type: ignore
import requests  # type: ignore
from loguru import logger
from pydantic import StringConstraints

import rigging as rg

SSH_HOST = "bandit.labs.overthewire.org"
SSH_PORT = 2220

MAX_PINS = 10
MAX_HISTORY = 5

SYSTEM_PROMPT = """\
You are a senior security expert with years of linux and CTF experience. You are solving a CTF which involves
locating a password on a server by executing commands via SSH.
"""

# Helpers

BASE_DOCS_URL = "https://overthewire.org/wargames/bandit"


def get_bandit_level_description(level: int) -> str:
    search = r"Level Goal</h2>(.+)<h2"
    response = requests.get(f"{BASE_DOCS_URL}/bandit{level}.html")
    response.raise_for_status()
    goal: str = re.findall(search, response.text, re.DOTALL)[0]
    goal = goal.replace("<p>", "").replace("</p>", "").strip()
    return re.sub("<.*?>", "", goal)


def connect_ssh(level: int, password: str) -> paramiko.SSHClient | None:
    username = f"bandit{level}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(SSH_HOST, SSH_PORT, username=username, password=password)
        logger.success(f"Connected to {SSH_HOST} on port {SSH_PORT} as {username}")
    except paramiko.AuthenticationException:
        logger.error(f"Failed to authenticate as {username} with password {password}")
        return None

    return client


def execute_ssh(client: paramiko.SSHClient, command: str, *, max_output_len: int = 5_000, timeout: int = 120) -> str:
    logger.debug(f"Executing:\n{command}")

    stdin, stdout, stderr = client.exec_command("/bin/bash", timeout=timeout)
    stdin.write(command + "\n")
    stdin.flush()
    stdin.channel.shutdown_write()

    time.sleep(1)

    try:
        stdout_output = stdout.read().decode(errors="backslashreplace")
        stderr_output = stderr.read().decode(errors="backslashreplace")
    except socket.timeout:
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
str_upper = t.Annotated[str, StringConstraints(to_upper=True)]


class Action(rg.Model, abc.ABC):
    def run(self, state: "State") -> str:
        raise NotImplementedError


class UpdateMyGoal(Action):
    goal: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return UpdateMyGoal(goal="My new goal").to_pretty_xml()

    def run(self, state: "State") -> str:
        user_input = input(f"\nModel wants to set goal to '{self.goal}'? (y/N): ")
        if user_input.lower() != "y":
            self.goal = input("What is the real goal?: ")
        logger.success(f"Updating goal to '{self.goal}'")
        state.goals.append(self.goal)
        return "Goal updated."


class SaveMemory(Action):
    key: str_strip = rg.attr()
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return SaveMemory(key="my-note", content="Lots of custom data\nKeep this for later.").to_pretty_xml()

    def run(self, state: "State") -> str:
        logger.success(f"Storing '{self.key}':\n{self.content}")
        state.memories[self.key] = self.content
        return f"Stored '{self.key}'."


class RecallMemory(Action):
    key: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return RecallMemory(key="last-thoughts").to_pretty_xml()

    def run(self, state: "State") -> str:
        value = state.memories.get(self.key, "Not found.")
        logger.success(f"Recalling '{self.key}'\n{value}")
        return value


class DeleteMemory(Action):
    key: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return DeleteMemory(key="my-note").to_pretty_xml()

    def run(self, state: "State") -> str:
        logger.success(f"Forgetting '{self.key}'")
        state.memories.pop(self.key, None)
        return f"Forgot '{self.key}'."


class PinToTop(Action):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return PinToTop(content="This is the auth token: 1234").to_pretty_xml()

    def run(self, state: "State") -> str:
        logger.success(f"Pinning '{self.content}'")
        state.pins.append(self.content)
        state.pins = state.pins[:MAX_PINS]
        return "Pinned."


class TryCommand(Action):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return TryCommand(content="whoami | grep abc").to_pretty_xml()

    def run(self, state: "State") -> str:
        return execute_ssh(state.client, self.content)


class SubmitPassword(Action):
    password: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return SubmitPassword(password="[long_pw_string]").to_pretty_xml()

    def run(self, state: "State") -> str:
        if re.search(r"[a-zA-Z0-9]{32}", self.password) is None:
            return "Invalid password format."

        next_level = state.level + 1
        client = connect_ssh(next_level, self.password)
        if client is None:
            return "Failed to connect. Invalid password."

        logger.success(f"Got password for level {next_level}: {self.password}")
        state.update_level(next_level, client=client)

        return f"Success! You are now on level {next_level}."


Actions = UpdateMyGoal | SaveMemory | RecallMemory | DeleteMemory | PinToTop | TryCommand | SubmitPassword
ActionsList: list[type[Actions]] = [
    UpdateMyGoal,
    SaveMemory,
    RecallMemory,
    DeleteMemory,
    PinToTop,
    TryCommand,
    SubmitPassword,
]


@dataclass
class State:
    # CTF
    client: paramiko.SSHClient = paramiko.SSHClient()
    level: int = 1
    level_details: str = ""

    # Core
    goals: list[str] = field(default_factory=list)
    next_actions: list[Actions] = field(default_factory=list)

    # Context
    pins: list[str] = field(default_factory=list)
    memories: dict[str, str] = field(default_factory=dict)
    history: list[tuple[Actions, str]] = field(default_factory=list)

    def update_level(
        self, level: int, *, password: str | None = None, client: paramiko.SSHClient | None = None
    ) -> None:
        if client is None and password is not None:
            self.client = connect_ssh(level, password)
        elif client is not None:
            self.client = client
        else:
            raise ValueError("Either password or client must be provided")

        self.level = level
        self.level_details = get_bandit_level_description(level + 1)
        self.goals.append(f"Find and submit the password to the next level ({self.level + 1})")

    def step(self) -> None:
        logger.info(f"Processing {len(self.next_actions)} action(s)")
        for action in self.next_actions:
            self.history.append((action, action.run(self)))
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
{UpdateMyGoal.xml_example()}

## Commands

To execute a command on the remote host via SSH:
{TryCommand.xml_example()}

## Password

When you have the password to the next level, provide it so the system can authenticate you to the next level:
{SubmitPassword.xml_example()}

## Examples

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

Output exactly 1 new action from the list above in your response. Prior action results are displayed above.
"""


# CLI  + Core


async def core_loop(level: int, password: str, generator_id: str, max_iterations: int, max_actions: int) -> None:
    logger.success(f"Starting bandit on level {level}")

    # Prepare our objects

    generator = rg.get_generator(generator_id)
    base_chat = generator.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}],
    ).with_(max_tokens=4096)

    client = connect_ssh(level, password)
    state = State(client=client)
    state.update_level(level, client=client)

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
        if len(parsed) > max_actions:
            logger.warning("Model provided multiple actions, taking just the first")

        state.next_actions = parsed[:max_actions]
        return None

    for i in range(1, max_iterations + 1):
        try:
            logger.info(f"iter {i}/{max_iterations}")

            chat = await base_chat.fork(state.get_prompt()).then(parse_actions).arun()
            logger.info(f"Last:\n{chat.last.content}")
            state.step()

        except KeyboardInterrupt:
            logger.info("Interrupted")
            check = input("\nSet a new goal? (y/N): ")
            if check.lower() == "y":
                new_goal = input("Enter new goal: ")
                state.goals.append(new_goal)
            else:
                raise

    # Final stats

    logger.info("bandit complete")

    logger.info("Goals:")
    for goal in state.goals:
        logger.info(f" |- {goal}")

    logger.info("Memories:")
    for key, value in state.memories.items():
        logger.info(f" |- {key}:\n{value}")


@click.command()
@click.option("-l", "--level", type=int, default=0, help="Starting level (1-34)")
@click.option("-p", "--password", type=str, default="bandit0", help="Starting password")
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
    default=5,
    help="Number of parallel agents",
)
@click.option(
    "-m",
    "--max-actions",
    type=int,
    default=3,
    help="Maximum number of actions allowed per generation",
)
@click.option(
    "--log-level",
    type=click.Choice(_shared.LogLevelList),
    default="info",
)
@click.option("--log-file", type=click.Path(path_type=pathlib.Path), default="bandit.log")
@click.option(
    "--log-file-level",
    type=click.Choice(_shared.LogLevelList),
    default="trace",
)
def cli(
    level: int,
    password: str,
    generator_id: str,
    max_iterations: int,
    parallel_agents: int,
    max_actions: int,
    log_level: str,
    log_file: pathlib.Path,
    log_file_level: _shared.LogLevelLiteral,
) -> None:
    _shared.configure_logging(log_level, log_file, log_file_level)
    asyncio.run(core_loop(level, password, generator_id, max_iterations, max_actions))


if __name__ == "__main__":
    cli()
