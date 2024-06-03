import abc
import asyncio
import json
import pathlib
import typing as t
from dataclasses import dataclass, field

import click
import httpx
from loguru import logger
from pydantic import StringConstraints

import rigging as rg
from rigging import logging

# Constants

MAX_PINS = 10
MAX_HISTORY = 5

SYSTEM_PROMPT = """\
You are a principal application security tester assessing vulnerabilities in web APIs.
"""

# Models

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]
str_upper = t.Annotated[str, StringConstraints(to_upper=True)]


class Action(rg.Model, abc.ABC):
    @abc.abstractmethod
    async def run(self, state: "State") -> str:
        ...


class Header(rg.Model):
    name: str = rg.attr()
    value: str_strip


class Parameter(rg.Model):
    name: str = rg.attr()
    value: str_strip


class Request(Action):
    method: str_upper = rg.attr()
    path: str = rg.attr()
    headers: list[Header] = rg.wrapped("headers", rg.element(default=[]))
    url_params: list[Parameter] = rg.wrapped("url_params", rg.element(default=[]))
    body: str_strip = rg.element(default="")

    @classmethod
    def xml_example(cls) -> str:
        return Request(
            method="GET",
            path="/$path",
            headers=[Header(name="X-Header", value="my-value")],
            url_params=[Parameter(name="name", value="test-param")],
            body="$body",
        ).to_pretty_xml()

    async def run(self, state: "State") -> str:
        response = await send_request(state.client, self)
        logger.success(f"{self.method} '{self.path}' -> {response.status_code}")
        state.traffic.append((self, response))
        return response.to_pretty_xml()


class Response(rg.Model):
    status_code: int = rg.attr()
    headers: list[Header] = rg.element(defualt=[])
    body: str_strip = rg.element(default="")


class UpdateGoal(Action):
    goal: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return UpdateGoal(goal="My new goal").to_pretty_xml()

    async def run(self, state: "State") -> str:
        user_input = input(f"\nModel wants to set goal to '{self.goal}'? (y/N): ")
        if user_input.lower() != "y":
            self.goal = input("What is the real goal? (empty for keep existing): ") or self.goal
        logger.success(f"Updating goal to '{self.goal}'")
        state.goals.append(self.goal)
        return "Goal updated."


class SaveMemory(Action):
    key: str_strip = rg.attr()
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return SaveMemory(key="my-note", content="Lots of custom data\nKeep this for later.").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"Storing '{self.key}':\n{self.content}")
        state.memories[self.key] = self.content
        return f"Stored '{self.key}'."


class RecallMemory(Action):
    key: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return RecallMemory(key="last-thoughts").to_pretty_xml()

    async def run(self, state: "State") -> str:
        value = state.memories.get(self.key, "Not found.")
        logger.success(f"Recalling '{self.key}'\n{value}")
        return value


class DeleteMemory(Action):
    key: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return DeleteMemory(key="my-note").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"Forgetting '{self.key}'")
        state.memories.pop(self.key, None)
        return f"Forgot '{self.key}'."


class PinToTop(Action):
    content: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return PinToTop(content="This is the auth token: 1234").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"Pinning '{self.content}'")
        state.pins.append(self.content)
        state.pins = state.pins[:MAX_PINS]
        return "Pinned."


class SetHeaderOnSession(Action):
    name: str_strip = rg.attr()
    value: str_strip

    @classmethod
    def xml_example(cls) -> str:
        return SetHeaderOnSession(name="X-Header", value="my-value").to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success(f"Adding header '{self.name}' with value '{self.value}'")
        state.client.headers[self.name] = self.value
        return "Header added."


class ResetSession(Action):
    @classmethod
    def xml_example(cls) -> str:
        return ResetSession().to_pretty_xml()

    async def run(self, state: "State") -> str:
        logger.success("Resetting session")
        state.client.headers.clear()
        return "Session reset."


Actions = t.Union[
    UpdateGoal,
    SaveMemory,
    RecallMemory,
    PinToTop,
    RecallMemory,
    DeleteMemory,
    Request,
    SetHeaderOnSession,
    ResetSession,
]
ActionsList: list[type[Action]] = [
    UpdateGoal,
    SaveMemory,
    RecallMemory,
    PinToTop,
    RecallMemory,
    DeleteMemory,
    Request,
    SetHeaderOnSession,
    ResetSession,
]


@dataclass
class State:
    # Required
    client: httpx.AsyncClient
    max_actions: int
    base_chat: rg.ChatPipeline

    # Core
    goals: list[str] = field(default_factory=list)
    next_actions: list[Actions] = field(default_factory=list)

    # Context
    pins: list[str] = field(default_factory=list)
    memories: dict[str, str] = field(default_factory=dict)
    traffic: list[tuple[Request, Response]] = field(default_factory=list)
    history: list[tuple[Actions, str]] = field(default_factory=list)

    async def step(self) -> None:
        logger.info(f"Processing {len(self.next_actions)} action(s)")
        for action in self.next_actions:
            self.history.append((action, await action.run(self)))
        self.next_actions.clear()

    def get_prompt(self, max_history: int = MAX_HISTORY) -> str:
        traffic = "\n".join(
            f"{request.method} {request.path} -> {response.status_code}"
            for request, response in self.traffic
            if response.status_code != 404
        )
        memories = "\n".join(self.memories.keys())
        previous_goals = "\n".join(self.goals[:-1] if len(self.goals) > 1 else [])
        current_goal = self.goals[-1]
        history = "\n---\n".join([h[0].to_pretty_xml() + "\n" + h[1] for h in self.history[-max_history:]])
        pinned = "\n".join(self.pins)
        headers = "\n".join(f"{k}: {v}" for k, v in self.client.headers.items())
        return f"""\
# Context

<previous-traffic count={len(self.traffic)}>
{traffic}
</previous-traffic>

<memories>
{memories or 'No memories yet.'}
</memories>

<last-{max_history}-actions>
{history or 'No actions taken yet'}
</last-{max_history}-actions>

<pinned>
{pinned or 'No pinned context yet.'}
</pinned>

<current-headers>
{headers or 'No headers set'}
</current-headers>

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

When you believe you've accomplished your current goal, pick a new one to continue your exploration.
{UpdateGoal.xml_example()}

## Web Traffic

To issue a web request, use the following format:
{Request.xml_example()}

To add a permanent header to all future requests (useful for auth tokens):
{SetHeaderOnSession.xml_example()}

To reset your current session and clear all headers:
{ResetSession.xml_example()}

# Guidance

- Focus on exploring as much of the application as possible.
- Do not use placeholder values in any of your actions.
- Recall your memories to help you make decisions.
- Pay attention to your previous traffic, avoid making unnecessary requests.
- Learned important information? Pin it for reference.
- Experiment frequently and be adjust your approach to make progress.
- All actions are useful, be organized and methodical in your approach.

Output up to {self.max_actions} new actions from the list above in your response. Prior action results are displayed above.
"""


# Functions


def format_http_request(request: httpx.Request) -> str:
    http_request = f"{request.method} {request.url} HTTP/1.1\n"
    http_request += "".join(f"{k}: {v}\n" for k, v in request.headers.items())
    if request.content:
        http_request += "\n" + request.content.decode("utf-8")
    return http_request


def format_http_response(response: httpx.Response) -> str:
    http_response = f"HTTP/1.1 {response.status_code} {response.reason_phrase}\n"
    http_response += "".join(f"{k}: {v}\n" for k, v in response.headers.items())
    if response.content:
        http_response += "\n" + response.text
    return http_response


async def send_request(client: httpx.AsyncClient, request: Request) -> Response:
    try:
        json_body = json.loads(request.body)
    except json.JSONDecodeError:
        json_body = None

    httpx_request = client.build_request(
        method=request.method,
        url=request.path,
        headers={header.name: header.value for header in request.headers},
        content=request.body if not json_body else None,
        json=json_body,
    )

    if not json_body:
        httpx_request.headers["Content-Type"] = "application/x-www-form-urlencoded"

    logger.trace(f"Request: \n{format_http_request(httpx_request)}")
    httpx_response = await client.send(httpx_request)
    logger.trace(f"Response:\n{format_http_response(httpx_response)}")

    return Response(
        status_code=httpx_response.status_code,
        headers=[Header(name=name, value=value) for name, value in httpx_response.headers.items()],
        body=httpx_response.text,
    )


# CLI  + Core


async def agent_loop(
    state: State,
    max_iterations: int,
) -> None:
    async def parse_actions(chat: rg.Chat) -> t.Optional[rg.Chat]:
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
            logger.warning(f"Model provided more actions than allows {len(parsed)} > {state.max_actions}")

        state.next_actions = parsed[: state.max_actions]
        return None

    for i in range(1, max_iterations + 1):
        logger.info(f"Iteration {i}/{max_iterations}")
        await state.base_chat.fork(state.get_prompt()).then(parse_actions).run()
        await state.step()


@click.command()
@click.option(
    "-G",
    "--first-goal",
    type=str,
    default="Find the API spec, register a user, get authenticated, then exploit.",
    help="First goal to perform",
)
@click.option(
    "-g",
    "--generator-id",
    type=str,
    default="anthropic/claude-3-sonnet-20240229",
    required=True,
    help="Rigging generator identifier (gpt-4, mistral/mistral-medium, etc.)",
)
@click.option(
    "-u",
    "--base-url",
    type=str,
    required=True,
    help="URL of the target application",
)
@click.option(
    "-p",
    "--proxy",
    type=str,
    help="HTTP proxy to use for requests",
)
@click.option(
    "-i",
    "--max-iterations",
    type=int,
    default=30,
    help="Maximum number of iterations",
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
@click.option("--log-file", type=click.Path(path_type=pathlib.Path), default="dvra.log")
@click.option(
    "--log-file-level",
    type=click.Choice(logging.LogLevelList),
    default="trace",
)
def cli(
    first_goal: str,
    generator_id: str,
    base_url: str,
    proxy: t.Optional[str],
    max_iterations: int,
    max_actions: int,
    log_level: logging.LogLevelLiteral,
    log_file: pathlib.Path,
    log_file_level: logging.LogLevelLiteral,
) -> None:
    """
    Rigging example for agentic exploitation of the Damn Vulnerable Restual API (DVRA).
    """

    logging.configure_logging(log_level, log_file, log_file_level)

    logger.success("Starting DVRA")

    # Prepare our objects

    generator = rg.get_generator(generator_id)
    client = httpx.AsyncClient(
        base_url=base_url,
        verify=False,
        proxies=(
            {
                "http://": proxy,
                "https://": proxy,
            }
            if proxy
            else None
        ),
    )

    base_chat: rg.ChatPipeline = generator.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}],
        rg.GenerateParams(max_tokens=4096),
    )

    state = State(client=client, max_actions=max_actions, base_chat=base_chat, goals=[first_goal])

    logger.info("Starting with '{}'", first_goal)

    while True:
        try:
            asyncio.run(agent_loop(state, max_iterations))
        except KeyboardInterrupt:
            logger.info("Interrupted")
            check = input("\nSet a new goal? (y/N): ")
            if check.lower() == "y":
                new_goal = input("Enter new goal: ")
                state.goals.append(new_goal)
            else:
                raise


if __name__ == "__main__":
    cli()
