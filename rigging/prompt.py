def system_tool_extension(call_format: str, tool_descriptions: str) -> str:
    return f"""\
# Tool Use
In this environment you have access to a set of tools you can use to improve your responses.

## Tool Call Format
{call_format}

## Available Tools
{tool_descriptions}

You can use any of the available tools by responding in the call format above. The XML will be parsed and the tool(s) will be executed with the parameters you provided. The results of each tool call will be provided back to you before you continue the conversation. You can execute multiple tool calls by continuing to respond in the format above until you are finished. Function calls take explicit values and are independent of each other. Tool calls cannot share, re-use, and transfer values between eachother. The use of placeholders is forbidden.

The user will not see the results of your tool calls, only the final message of your conversation. Wait to perform your full response until after you have used any required tools. If you intend to use a tool, please do so before you continue the conversation.
"""
