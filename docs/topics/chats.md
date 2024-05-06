!!! note
    This content is currently being refactored

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