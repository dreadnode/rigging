# Rigging

Rigging is a lightweight LLM interaction framework built on Pydantic XML and LiteLLM. It supports useful primitives for validating LLM output and adding tool calling abilities to models that don't natively support it. It also has various helpers for common tasks like structured object parsing, templating chats, overloading generation parameters, stripping chat segments, and continuing conversations.

Modern python with type hints, pydantic validation, native serialization support, etc.

```
pip install rigging
```

### Overview

The basic flow in rigging is:

1. Get a generator object
2. Call `.chat()` to produce a `PendingChat`
3. Call `.run()` on a `PendingChat` to get a `Chat`

`PendingChat` objects hold any messages waiting to be delivered to an LLM in exchange
for a new response message. Afterwhich it is converted into a `Chat` which holds
all messages prior to generation (`.prev`) and after generation (`.next`).

You should think of `PendingChat` objects like the configurable pre-generation step
with calls like `.overload()`, `.apply()`, `.until()`, `.using()`, etc. Once you call
`.run()` the generator is used to produce the next message based on the prior context
and any constraints you have in place. Once you have a `Chat` object, the interation
is "done" and you can inspect/parse the messages.

You'll often see us use functional styling chaining as most of our
utility functions return the object back to you.

```python
chat = generator.chat(...).using(...).until(...).overload(...).run()
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

