# Rigging

Rigging is a lightweight LLM interaction framework built on Pydantic XML and LiteLLM. It supports useful primitives for validating LLM output and adding tool calling abilities to models that don't natively support it. It also has various helpers for common tasks like structured object parsing, templating chats, overloading generation parameters, stripping chat segments, and continuing conversations.

Modern python with type hints, pydantic validation, native serialization support, etc.

### Basic Chats

```python
import rigging as rg

generator = rg.get_generator("claude-2.1")
chat = generator.chat(
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
).run()

print(chat.last)
# [assistant]: Hello!

print(f"{chat.last!r}")
# Message(role='assistant', parts=[], content='Hello!')

print(chat.prev)
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
# ]

print(chat.json)
# [{ ... }]

```

### Model Parsing

```python
import rigging as rg

class Answer(rg.Model):
    content: str

chat = (
    rg.get_generator("claude-2.1")
    .chat([
        {"role": "user", "content": f"Say your name between {Answer.xml_tags()}."},
    ])
    .until_parsed_as(Answer)
    .run()
)

answer = chat.last.parse(Answer)
print(answer.content)

# "Claude"

print(f"{chat.last!r}")

# Message(role='assistant', parts=[
#   ParsedMessagePart(model=Answer(content='Claude'), ref='<answer>Claude</answer>')
# ], content='<Answer>Claude</Answer>')

chat.last.content = "new content" # Updating content strips parsed parts
print(f"{chat.last!r}")

# Message(role='assistant', parts=[], content='new content')
```

### Complex Models

```python
import rigging as rg

class Inner(rg.Model):
    type: str = rg.attr()
    content: str

class Outer(rg.Model):
    name: str = rg.attr()
    inners: list[Inner] = rg.element()

outer = Outer(name="foo", inners=[
    Inner(type="cat", content="meow"),
    Inner(type="dog", content="bark")
])

print(outer.to_pretty_xml())

# <outer name="foo">
#    <inner type="cat">meow</inner>
#    <inner type="dog">bark</inner>
# </outer>
```

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
    .using(WeatherTool())
    .run()
)

# [=] get_for_city('London')

print(chat.last.content)

# "Based on the information I've received, the weather in London is nice today."
```

### Continuing Chats

```python
import rigging as rg

generator = rg.get_generator("gpt-3.5-turbo")
chat = generator.chat([
        {"role": "user", "content": "Hello, how are you?"},
]).run()

print(chat.last.content)

# "Hello! I'm an AI language model, ..."

cont = chat.continue_(
    {"role": "user", "content": "That's good, tell me a joke"}
).run()

print(cont.last.content)

# "Sure, here's a joke for you: ..."
```

### Basic Templating

```python
import rigging as rg

template = rg.get_generator("gpt-4").chat([
    {"role": "user", "content": "What is the capitol of $country?"},
])

for country in ["France", "Germany"]:
    print(template.apply(country=country).run().last)

# The capital of France is Paris.
# The capital of Germany is Berlin.
```

### Overload Generation Params

```python
import rigging as rg

pending = rg.get_generator("gpt-3.5-turbo,max_tokens=50").chat([
    {"role": "user", "content": "Say a haiku about boats"},
])

for temp in [0.1, 0.5, 1.0]:
    print(pending.overload(temperature=temp).run().last.content)

```

### Strip Parsed Sections

```python
import rigging as rg

class Reasoning(rg.Model):
    content: str

meaning = rg.get_generator("claude-2.1").chat([
    {
        "role": "user",
        "content": "What is the meaning of life in one sentence? "
        f"Document your reasoning between {Reasoning.xml_tags()} tags.",
    },
]).run()

# Gracefully handle mising models
reasoning = meaning.last.try_parse(Reasoning)
if reasoning:
    print("reasoning:", reasoning.content.strip())

# Strip parsed content to avoid sharing
# previous thoughts with the model.
without_reasons = meaning.strip(Reasoning)
print("meaning of life:", without_reasons.last.content.strip())

# follow_up = without_thoughts.continue_(...)
```

### Custom Generator

Any custom generator simply needs to implement a `complete` function, and 
then it can be used anywhere inside rigging.

```python
class Custom(Generator):
    # model: str
    # api_key: str
    # params: GeneratorParams
    
    custom_field: bool

    def complete(
        self,
        messages: t.Sequence[rg.Message],
        overloads: GenerateParams = GenerateParams(),
    ) -> rg.Message:
        # Access self vars where needed
        api_key = self.api_key
        model_id = self.model

        # Merge in args for API overloads
        marged: dict[str, t.Any] = self._merge_params(overloads)

        # response: str = ...

        return rg.Message("assistant", response)


generator = Custom(model='foo', custom_field=True)
generator.chat(...)
```

*Note: we currently don't have anyway to "register" custom generators for `get_generator`.*

### Logging

By default rigging disables it's logger with loguru. To enable it run:

```python
from loguru import logger

logger.enable('rigging')
```

To configure loguru terminal + file logging format overrides:

```python
from rigging.logging import configure_logging

configure_logging(
    'info',      # stderr level
    'out.log',   # log file (optional)
    'trace'      # log file level
)
```
*(This will remove existing handlers, so you might prefer to configure them yourself)*