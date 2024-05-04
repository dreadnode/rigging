# Rigging

Rigging is a lightweight LLM interaction framework built on Pydantic XML. The goal is to make leveraging LLMs in production pipelines as simple and effictive as possible. Here are the highlights:

- **Structured Pydantic models** can be used interchangably with unstructured text output.
- LiteLLM as the default generator giving you **instant access to a huge array of models**.
- Add easy **tool calling** abilities to models which don't natively support it.
- Store different models and configs as **simple connection strings** just like databases.
- Chat templating, forking, continuations, generation parameter overloads, stripping segments, etc.
- Modern python with type hints, async support, pydantic validation, serialization, etc.

```py
import rigging as rg
from rigging.model import CommaDelimitedAnswer as Answer

answer = rg.get_generator('gpt-4') \
    .chat(f"Give me 3 famous authors between {Answer.xml_tags()} tags.") \
    .until_parsed_as(Answer) \
    .run()

answer = chat.last.parse(Answer)
print(answer.items)

# ['J. R. R. Tolkien', 'Stephen King', 'George Orwell']
```

Rigging is built and maintained by [dreadnode](https://dreadnode.io) where we use it daily for our work.

## Installation
We publish every version to Pypi:
```bash
pip install rigging
```

If you want to build from source:
```bash
cd rigging/
poetry install
```

## Workflow

1. Get a [`Generator`][rigging.generator.Generator] object - usually with [`get_generator()`][rigging.generator.get_generator].
2. Call [`generator.chat()`][rigging.generator.Generator.chat] to produce a [`PendingChat`][rigging.chat.PendingChat] and ready it for generation.
3. Call [`pending.run()`][rigging.chat.PendingChat.run] to kick off generation and get your final [`Chat`][rigging.chat.Chat] object.

[`PendingChat`][rigging.chat.PendingChat] objects hold any messages waiting to be delivered to an LLM in exchange
for a new response message. These objects are also where most of the power in rigging comes from. You'll build a
generation pipeline with options, parsing, callbacks, etc. After prep this pending chat is converted into a 
final [`Chat`][rigging.chat.Chat] which holds all messages prior to generation ([`.prev`][rigging.chat.Chat.prev]) 
and after generation ([`.next`][rigging.chat.Chat.next]).

You should think of [`PendingChat`][rigging.chat.PendingChat] objects like the configurable pre-generation step
with calls like [`.overload()`][rigging.chat.PendingChat.overload], [`.apply()`][rigging.chat.PendingChat.apply], 
[`.until()`][rigging.chat.PendingChat.until], [`.using()`][rigging.chat.PendingChat.using], etc. Once you call one
of the many [`.run()`][rigging.chat.PendingChat.run] functions, the generator is used to produce the next 
message (or many messages) based on the prior context and any constraints you have in place. Once you have a 
[`Chat`][rigging.chat.Chat] object, the interation is "done" and you can inspect and operate on the messages.

You'll often see us use functional styling chaining as most of our
utility functions return the object back to you.

```python
chat = generator.chat(...) \ (1)
    .using(...).until(...).overload(...) \
    .run()
```

### Continuing Chats

```python
import rigging as rg

generator = rg.get_generator("gpt-3.5-turbo")
chat = generator.chat([
        {"role": "user", "content": "Hello, how are you?"},
])

# We can fork (continue_) before generation has occured
specific = chat.fork("Be specific please.").run()
poetic = chat.fork("Be as poetic as possible").overload(temperature=1.5).run()

# We can also fork (continue_) after generation
next_chat = poetic.fork(
    {"role": "user", "content": "That's good, tell me a joke"}
)

update = next_chat.run()
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

