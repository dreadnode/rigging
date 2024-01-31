def system_tool_extension(call_format: str, tool_descriptions: str) -> str:
    return f"""\
In this environment you have access to a set of tools you can use to improve your responses.

Tool call format:
{call_format}

Available tools:
{tool_descriptions}

You can use tools by responding in the format above. The inputs will be parsed and the specified tool function will be executed with the parameters you provided. The results of each function call will be given before you continue the conversation. You can execute multiple steps of function calls by continuing to respond in the format above. Function calls take explicit values and are independent of each other. Function calls CANNOT use the results of other functions. DO NOT USE terms like `$result` or `TOOL_RESULT` in your parameters.

The user will not see the results of your tool calls, only the final message of your conversation. Wait to perform your full response until after you have used any required tools.
"""
