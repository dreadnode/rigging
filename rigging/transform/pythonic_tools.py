import ast
import contextlib
import itertools
import json
import typing as t
import uuid

from pydantic_xml import create_model as pydantic_xml_create_model

from rigging.generator import GenerateParams
from rigging.message import (
    Message,
    inject_system_content,
    strip_system_content,
)
from rigging.tools.base import FunctionCall, ToolCall, ToolDefinition, ToolResponse
from rigging.transform.base import PostTransform, Transform

if t.TYPE_CHECKING:
    from rigging.chat import Chat


class PythonicToolAstError(Exception):
    pass


def _get_parameter_value(val: ast.expr) -> t.Any:
    """Recursively parses an AST expression node into a Python literal."""
    if isinstance(val, ast.Constant):
        return val.value
    if isinstance(val, ast.Dict):
        if not all(isinstance(k, ast.Constant) for k in val.keys):
            raise PythonicToolAstError("Dict tool call arguments must have literal keys")
        return {
            k.value: _get_parameter_value(v)  # type: ignore[union-attr]
            for k, v in zip(val.keys, val.values, strict=True)
        }
    if isinstance(val, ast.List):
        return [_get_parameter_value(v) for v in val.elts]
    if isinstance(val, ast.Name):
        return val.id
    if isinstance(val, ast.Attribute):
        return ast.unparse(val)

    raise PythonicToolAstError(f"Tool call arguments must be literals, but got {type(val)}")


def _ast_call_to_rigging_tool_call(call: ast.Call) -> ToolCall:
    """Converts a single ast.Call node to a rigging.tools.ToolCall."""
    if not isinstance(call.func, ast.Name):
        raise PythonicToolAstError("Invalid tool call name: must be a simple name.")

    function_name = call.func.id
    arguments = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise PythonicToolAstError("Tool call arguments must be keyword arguments.")
        arguments[keyword.arg] = _get_parameter_value(keyword.value)

    return ToolCall(
        id=f"rg-{uuid.uuid4().hex[:8]}",
        function=FunctionCall(
            name=function_name,
            arguments=json.dumps(arguments, ensure_ascii=False),
        ),
    )


def _extract_bracketed_blocks(text: str) -> list[str]:
    """Finds and extracts all top-level bracketed [...] blocks from a string."""
    candidates = []
    start_index = -1
    depth = 0
    for i, char in enumerate(text):
        if char == "[":
            if depth == 0:
                start_index = i
            depth += 1
        elif char == "]":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_index != -1:
                    candidates.append(text[start_index : i + 1])
                    start_index = -1
    return candidates


def _attempt_parse_tool_calls_from_string(candidate_str: str) -> list[ToolCall] | None:
    """Attempts to parse a string as a list of valid tool calls."""
    with contextlib.suppress(Exception):
        module = ast.parse(candidate_str, mode="eval")
        if not isinstance(module.body, ast.List):
            raise PythonicToolAstError("Tool output must be a list of function calls.")
        if not all(isinstance(element, ast.Call) for element in module.body.elts):
            raise PythonicToolAstError("All elements in tool list must be function calls.")
        return [
            _ast_call_to_rigging_tool_call(call)
            for call in module.body.elts
            if isinstance(call, ast.Call)
        ]
    return None


def _render_tool_call_to_pythonic_string(tool_call: ToolCall) -> str:
    """Renders a single ToolCall object to its pythonic string representation."""
    args_dict = json.loads(tool_call.function.arguments)
    args_str = ", ".join(f"{key}={json.dumps(value)}" for key, value in args_dict.items())
    return f"{tool_call.function.name}({args_str})"


def pythonic_tools_prompt(tools: list[ToolDefinition]) -> str:
    tools_str = "\n".join(tool.function.model_dump_json() for tool in tools)
    return f"""\
# Tools

You may call one or more functions to assist with the user query. \
Don't make assumptions about what values to plug into functions.

<available-tools>
{tools_str}
</available-tools>

If you decide to invoke one or more of the tools, \
you must respond with a list of function calls in python code syntax like so:

```
[func_name1(params_name1=params_value1, params_name2=params_value2, ...), func_name2(params)]
```

All function calls must be part of a single list.\
"""


def make_tools_to_pythonic_transform(
    *,
    system_tool_prompt: t.Callable[[list[ToolDefinition]], str] | str | None = None,
    tool_responses_as_user_messages: bool = True,
    tool_response_tag: str = "tool-response",
) -> Transform:
    """
    Create a transform that converts tool calls to a pythonic list format.

    This transform will:
    1. Inject a system prompt with tool definitions serialized as JSON.
    2. Convert existing tool calls in messages to `[my_func(arg=...)]` format.
    3. Convert tool result messages into `<tool-response>` blocks in a user message (optional).
    4. In the post-transform, parse the model's output using a robust,
       AST-based parser to extract tool calls from the generated string.

    Args:
        system_tool_prompt: A callable or string that generates the system prompt for tools.
        tool_responses_as_user_messages: If True, tool responses will be converted to user messages wrapped in tool response tags.
        tool_response_tag: The tag to use for tool responses in user messages.

    Returns:
        A transform function that processes messages and generate params.
    """

    system_tool_prompt = system_tool_prompt or pythonic_tools_prompt

    tool_response_cls = pydantic_xml_create_model(
        "ToolResponse",
        __base__=ToolResponse,
        __cls_kwargs__={"tag": tool_response_tag},
        __tag__=tool_response_tag,
    )

    async def tools_to_pythonic_transform(
        messages: list[Message],
        params: GenerateParams,
    ) -> tuple[list[Message], GenerateParams, PostTransform | None]:
        # Inject tool definitions into the system prompt

        system_prompt = (
            system_tool_prompt
            if isinstance(system_tool_prompt, str)
            else system_tool_prompt(params.tools or [])
        )
        messages = inject_system_content(messages, system_prompt)

        # Render existing tool calls and responses

        updated_messages: list[Message] = []

        for is_tool_group, message_group in itertools.groupby(
            messages, key=lambda m: tool_responses_as_user_messages and m.role == "tool"
        ):
            if is_tool_group:
                user_message = Message(role="user", content="")
                for tool_message in message_group:
                    user_message.append_slice(
                        tool_response_cls(
                            id=tool_message.tool_call_id or "",
                            result=tool_message.content,
                        ),
                        "tool_response",
                        metadata={"id": tool_message.tool_call_id or ""},
                    )
                updated_messages.append(user_message)
                continue

            for message in message_group:
                if not message.tool_calls:
                    updated_messages.append(message)
                    continue

                updated_message = message.clone()
                rendered_calls = [
                    _render_tool_call_to_pythonic_string(tc) for tc in message.tool_calls
                ]
                updated_message.tool_calls = None
                updated_message.append_slice(
                    f"[{', '.join(rendered_calls)}]",
                    "tool_call",
                    metadata={
                        "id": message.tool_calls[0].id or ""
                    },  # TODO(nick): Handle multiple tool call slices
                )
                updated_messages.append(updated_message)

        # Save any existing tool params

        existing_tool_definitions = params.tools
        params.tools = None
        existing_tool_choice = params.tool_choice
        params.tool_choice = None

        # Build post transform

        async def pythonic_to_tools_transform(chat: "Chat") -> "Chat":
            # Convert the tool calls and strip them

            for message in [m for m in chat.all if m.role == "assistant"]:
                # Restore original tool calls - fast path for efficiency and consistency

                for slice_ in message.slices:
                    if slice_.type == "tool_call" and isinstance(slice_.obj, ToolCall):
                        message.tool_calls = message.tool_calls or []
                        message.tool_calls.append(slice_.obj)
                        message.remove_slices(slice_)

                # Otherwise, find any new tool calls in the content

                candidates = _extract_bracketed_blocks(message.content)
                parsed_results: list[tuple[str, list[ToolCall]]] = []
                for candidate_str in candidates:
                    if parsed_calls := _attempt_parse_tool_calls_from_string(candidate_str):
                        parsed_results.append((candidate_str, parsed_calls))  # noqa: PERF401

                if not parsed_results:
                    continue

                # NOTE(nick): We only take the last successfully parsed block
                tool_calls_str, tool_calls = parsed_results[-1]
                message.tool_calls = tool_calls
                message.remove_slices()
                message.content = message.content.replace(tool_calls_str, "").strip()

            # Convert our tool responses

            updated_messages = []
            for message in messages:
                if message.role != "user" or not (
                    tool_responses := message.try_parse_set(tool_response_cls)
                ):
                    updated_messages.append(message)
                    continue

                for tool_response in tool_responses:
                    updated_messages.append(  # noqa: PERF401
                        Message(
                            role="tool",
                            content=tool_response.result,
                            tool_call_id=tool_response.id,
                        )
                    )

            # Restore the params

            chat.params = chat.params or GenerateParams()
            chat.params.tools = existing_tool_definitions
            chat.params.tool_choice = existing_tool_choice

            # Strip the system prompt content

            chat.messages = strip_system_content(updated_messages, system_prompt)

            return chat

        return updated_messages, params, pythonic_to_tools_transform

    return tools_to_pythonic_transform


tools_to_pythonic_transform = make_tools_to_pythonic_transform()
"""
A transform that converts tool calls to a pythonic list format.

See `make_tools_to_pythonic_transform` for more details and more behavior options.
"""
