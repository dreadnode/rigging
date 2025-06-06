import json
import typing as t

from rigging.generator import GenerateParams
from rigging.message import ContentText, Message, inject_system_content, strip_system_content
from rigging.transform.base import PostTransform, Transform

if t.TYPE_CHECKING:
    from rigging.chat import Chat


LLAMA_PROMPT = """\
You have access to the following functions. To call a function, please \
respond with JSON for a function call. Respond in the format \
{"name": function name, "parameters": dictionary of argument name and its value}. \
Do not use variables.
"""


def make_tools_to_json_transform(
    *,
    system_tool_prompt_prefix: str = LLAMA_PROMPT,
) -> Transform:
    """
    Create a transform that converts tool calls to JSON format
    and tool responses to ipython role for Llama tokenization.
    """

    async def tools_to_json_transform(
        messages: list[Message],
        params: GenerateParams,
    ) -> tuple[list[Message], GenerateParams, PostTransform | None]:
        system_prompt_content = system_tool_prompt_prefix
        for tool in params.tools or []:
            system_prompt_content += tool.model_dump_json(indent=4) + "\n\n"

        messages = inject_system_content(messages, system_prompt_content)

        for message in messages:
            if message.tool_calls:
                formatted = "\n".join(
                    [
                        json.dumps(
                            {
                                "name": tool_call.function.name,
                                "parameters": json.loads(tool_call.function.arguments),
                            },
                        )
                        for tool_call in message.tool_calls
                    ],
                )
                message.content_parts.append(
                    ContentText(text=formatted),
                )
                message.tool_calls = None

        async def json_to_tools_transform(chat: "Chat") -> "Chat":
            # Restore original tool calls and roles
            for message in chat.all:
                if message.metadata and "_original_tool_calls" in message.metadata:
                    # Restore tool calls
                    message.tool_calls = message.metadata.pop("_original_tool_calls")
                    # Remove JSON content we added
                    lines = message.content.split("\n")
                    json_lines = len(message.tool_calls)
                    message.content = "\n".join(lines[:-json_lines]).rstrip()

                elif message.metadata and "_original_role" in message.metadata:
                    # Restore tool role
                    message.role = message.metadata.pop("_original_role")
                    message.tool_call_id = message.metadata.pop("_tool_call_id", None)

            chat.messages = strip_system_content(chat.messages, system_prompt_content)

            return chat

        return messages, params, json_to_tools_transform

    return tools_to_json_transform


tools_to_json_transform = make_tools_to_json_transform()
