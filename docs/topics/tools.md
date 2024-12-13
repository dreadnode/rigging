# Tools

Rigging supports the concept of tools through 2 implementations:

- **'API' Tools**: These are API-level tool definitions which require a support from a model provder.
- **'Native' Tools**: These are internally defined, parsed, and handled by Rigging (the original implementation).

In most cases, users should opt for API tools with better provider integrations and performance.

Regardless of tool type, the [`ChatPipeline.using()`][rigging.chat.ChatPipeline.using] method should be
used to register tools for use during generation.

=== "API Tools"

    ```py
    from typing import Annotated
    import requests
    import rigging as rg
 
    def get_weather(city: Annotated[str, "The city name to get weather for"]) -> str:
       "A tool to get the weather for a location"
       try:
          city = city.replace(" ", "+")
          return requests.get(f"http://wttr.in/{city}?format=2").text
       except:
          return "Failed to call the API"
 
    chat = (
       await 
       rg.get_generator("gpt-4o-mini")
       .chat("How is the weather in london?")
       .using(get_weather)
       .run()
    )

    # [user]: How is the weather in london?

    # [assistant]: 
    # |- get_weather({"city":"London"})
 
    # [tool]: 🌦 🌡️+6°C 🌬️↘35km/h
 
    # [assistant]: The weather in London is currently overcast with light rain ...
    ```

=== "Native Tools"

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
    
    chat = (
       await 
       rg.get_generator("gpt-4o-mini")
       .chat("How is the weather in london?")
       .using(WeatherTool())
       .run()
    )
    ```


## API Tools

API tools are defined as standard callables (async supported) and get wrapped in the 
[`rg.ApiTool`][rigging.tool.ApiTool] class before being used during generation.

We use Pydantic to introspect the callable and extract schema information from the signature with some great benefits:

1. API-compatible schema information from any function
2. Robust argument validation for incoming inference data
3. Flexible type handling for BaseModels, Fields, TypedDicts, and Dataclasses

Just after the tool is converted, we take the function schema and add it to the 
[GenerateParams.tools][rigging.generator.GenerateParams] inside the `ChatPipeline`.

```py
from typing_extensions import TypedDict
from typing import Annotated
from pydantic import Field
import rigging as rg

class Filters(TypedDict):
    city: Annotated[str | None, Field(description="The city to filter by")]
    age: int | None

def lookup_person(name: Annotated[str, "Full name"], filters: Filters) -> str:
    "Search the database for a person"
    ...


tool = rg.ApiTool(lookup_person)

print(tool.name)
# lookup_person

print(tool.description)
# Search the database for a person

print(tool.schema)
# {'$defs': {'Filters': {'properties': ...}
```

Internally, we leverage [`ChatPipeline.then()`][rigging.chat.ChatPipeline.then] to handle responses from the model and
attempt to resolve tool calls before starting another generation loop. This means that when you pass the tool function
into your chat pipeline will define it's order amongst other callbacks like [`.then()`][rigging.chat.ChatPipeline.then]
and [`.map()`][rigging.chat.ChatPipeline.map]

## Native Tools

Much like models, native tools inherit from a base [`rg.Tool`][rigging.tool.Tool] class. These subclasses are required
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

Integrating native tools into the generation process is as easy as passing an instantiation
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

    
### Under the Hood

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