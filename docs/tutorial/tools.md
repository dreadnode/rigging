### Tools

```python
from typing import Annotated
import rigging as rg

class WeatherTool(rg.Tool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "A tool to get the weather for a location"

    def get_for_city(self, city: Annotated[str, "The city name to get weather for"]) -> str:
        print(f"[=] get_for_city('{city}')")
        return f"The weather in {city} is nice today"

chat = (
    rg.get_generator("mistral/mistral-tiny")
    .chat(
        [
            {"role": "user", "content": "What is the weather in London?"},
        ]
    )
    .using(WeatherTool(), force=True)
    .run()
)

# [=] get_for_city('London')

print(chat.last.content)

# "Based on the information I've received, the weather in London is nice today."
```