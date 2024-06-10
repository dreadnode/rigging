# Tools

By popular demand, Rigging includes a basic helper layer to provide the concept of "Tools" to a model. It's
debatable whether this approach (or more specifically the way we present it to narrative models) is a good idea,
but it's a fast way to extend the capability of your generation into arbitrary code functions that you define.

## Writing Tools

Much like models, tools inherit from a base [`rg.Tool`][rigging.tool.Tool] class. These subclasses are required
to provide at least 1 function along with a name and description property to present to the LLM during generation.

Every function you define and the parameters within are required to carry both type hints and annotations that
describe their function.

```py
from typing import Annotated
import requests
import rigging as rg

class WeatherTool(rg.Tool):
    name = "weather"
    description = "A tool to get the weather for a location"

    def get_for_city(self, city: Annotated[str, "The city name to get weather for"]) -> str:
        try:
            city = city.replace(" ", "+")
            return requests.get(f"http://wttr.in/{city}?format=2").text
        except:
            return "Failed to call the API"
```

Integrating tools into the generation process is as easy as passing an instantiation
of your tool class to the [`ChatPipeline.using()`][rigging.chat.ChatPipeline.using] method.

```py
chat = (
   await
   rg.get_generator("gpt-3.5-turbo")
   .chat("What is the weather in London?")
   .using(WeatherTool(), force=True) # (1)!
   .run()
)

print(chat.last.content)
# The current weather in London is 57°F with a light breeze of 2mph.
```

1. The use of `force=True` here is optional, but results in the internal generation
   ensuring at least one tool is called before the generation completes.

If/when the LLM elects to emit a valid tool call in Riggings format, it will
side-step, process the arguments, ensure they conform to your function spec,
and execute the desired function. Results will be injected back into the chat
and the final message which does not include any tool calls will trigger the end
of the generation process.

??? tip "Tool State"

    It's worth noting that tools are passed as instantiated classes into Rigging,
    which means your tool is free to carry state about it's operations as time
    progresses. Whether this is a good software design decision is up to you.

    
## Under the Hood

If you are curious what is occuring "under the hood" (as you should), you can
print the entire conversation text and see our injected system prompt of
instructions for using a tool, along with the auto-generated XML description
of the `WeatherTool` we supplied to the model

```xml
[system]: # Tool Use
In this environment you have access to a set of tools you can use to improve your responses.

## Tool Call Format
<tool-calls>
   <tool-call tool="$TOOL_A" function="$FUNCTION_A">
      <parameter name="$PARAMETER_NAME" />
   </tool-call>
   <tool-call tool="$TOOL_B" function="$FUNCTION_B">
      <parameter name="$PARAMETER_NAME" />
   </tool-call>
</tool-calls>

## Available Tools
<tool-description-list>
   <tool-description name="weather" description="A tool to get the weather for a location">
      <functions>
         <tool-function name="get_for_city" description="">
            <parameters>
               <tool-parameter name="city" type="str" description="The city name to get weather for" />
            </parameters>
         </tool-function>
      </functions>
   </tool-description>
</tool-description-list>

You can use any of the available tools by responding in the call format above. The XML will be
parsed and the tool(s) will be executed with the parameters you provided. The results of each
tool call will be provided back to you before you continue the conversation. You can execute
multiple tool calls by continuing to respond in the format above until you are finished.
Function calls take explicit values and are independent of each other. Tool calls cannot share,
re-use, and transfer values between eachother. The use of placeholders is forbidden.

The user will not see the results of your tool calls, only the final message of your conversation.
Wait to perform your full response until after you have used any required tools. If you intend to
use a tool, please do so before you continue the conversation.


[user]: What is the weather in London?

[assistant]: <tool-calls>
   <tool-call tool="weather" function="get_for_city">
      <parameter name="city">London</parameter>
   </tool-call>
</tool-calls>

[user]: <tool-results>
   <tool-result tool="weather" function="get_for_city" error="false">
        &#9728;&#65039;   &#127777;&#65039;+57&#176;F &#127788;&#65039;&#8594;2mph
    </tool-result>
</tool-results>


[assistant]: The current weather in London is 57°F with a light breeze of 2mph.
```

Every tool assigned to the `ChatPipeline` will be processed by calling [`.get_description()`][rigging.tool.Tool.get_description]
and a minimal tool-use prompt will be injected as, or appended to, the system message.

!!! warning "The Curse of Complexity"

    Everything we add to the context window of a model introduces variance to it's outputs.
    Even the way we name parameters and tools can have a large impact on whether a model
    elects to output a tool call and how frequently or late it does so. For this reason
    tool calling in Rigging might not be the best way to accomplish your goals.

    Consider different approaches to your problem like isolating fixed input/output
    pairs and building a dedicated generation process around those, or pushing the
    model to think more about selecting from a series of "actions" it should take
    rather than "tools" it should use are part of a conversation.

    You also might consider a pipeline where incoming messages are scanned against
    a list of possible tools, and fork the generation process with something like
    [`.then`][rigging.chat.ChatPipeline.then] instead. 