# Chats and Messages

[`Chat`][rigging.chat.Chat] objects hold a sequence of [`Message`][rigging.message.Message] objects pre and post generation. This
is the most common way that we interact with LLMs, and the interface of both these and [`PendingChat`][rigging.chat.PendingChat]'s are
very flexible objects that let you tune the generation process, gather structured outputs, validate parsing, perform text replacements,
serialize and deserialize, fork conversations, etc.

## Basic Usage

```py
import rigging as rg

generator = rg.get_generator("claude-2.1")
chat = generator.chat(
    [
        {"role": "system", "content": "You're a helpful assistant."},
        {"role": "user", "content": "Say hello!"},
    ]
).run()

print(chat.last)
# [assistant]: Hello!

print(f"{chat.last!r}")
# Message(role='assistant', parts=[], content='Hello!')

print(chat.prev)
# [
#   Message(role='system', parts=[], content='You're a helpful assistant.'),
#   Message(role='user', parts=[], content='Say hello!'),
# ]

print(chat.message_dicts)
# [
#   {'role': 'system', 'content': 'You're a helpful assistant.'},
#   {'role': 'user', 'content': 'Say Hello!'},
#   {'role': 'assistant', 'content': 'Hello!'}
# ]

print(chat.conversation)
# [system]: You're a helpful assistant.

# [user]: Say hello!

# [assistant]: Hello!
```

## Templating (apply)

You can use both [`PendingChat.apply()`][rigging.chat.PendingChat.apply] and [`PendingChat.apply_to_all()`][rigging.chat.PendingChat.apply_to_all]
to swap values prefixed with `$` characters inside message contents for fast templating support.

This functionality uses [string.Template.safe_substitute](https://docs.python.org/3/library/string.html#string.Template.safe_substitute) underneath.

```py
import rigging as rg

template = rg.get_generator("gpt-4").chat([
    {"role": "user", "content": "What is the capitol of $country?"},
])

for country in ["France", "Germany"]:
    print(template.apply(country=country).run().last)

# The capital of France is Paris.
# The capital of Germany is Berlin.
```

## Parsed Parts

Message objects hold all of their parsed [`ParsedMessagePart`][rigging.message.ParsedMessagePart]'s inside their
[`.parts`][rigging.chat.Message.parts] property. These parts maintain both the instance of the parsed Rigging
model object and a [`.slice_`][rigging.message.ParsedMessagePart.slice_] property that defines exactly
where in the message content they are located.

Every time parsing occurs, these parts are re-synced by using [`.to_pretty_xml()`][rigging.model.Model.to_pretty_xml]
on the model, and stitching the clean content back into the message, fixing any other slices which might
have been affected by the operation, and ordering the [`.parts`][rigging.chat.Message.parts] property based on where
they occur in the message content.

```py
import rigging as rg
from pydantic import StringConstraints
from typing import Annotated

str_strip = Annotated[str, StringConstraints(strip_whitespace=True)]

class Summary(rg.Model):
    content: str_strip

message = rg.Message(
    "assistant",
    "Sure, the summary is: <summary  > Rigging is a very powerful library </summary>. I hope that helps!"
)

message.parse(Summary)

print(message.content) # (1)!
# Sure, the summary is: <summary>Rigging is a very powerful library</summary>. I hope that helps!

print(message.parts)
# [
#   ParsedMessagePart(model=Summary(content='Rigging is a very powerful library'), slice_=slice(22, 75, None))
# ]

print(message.content[message.parts[0].slice_])
# <summary>Rigging is a very powerful library</summary>
```

1. Notice how our message content got updated to reflect fixing the the extra whitespace
   in our start tag and our string stripping annotation.

## Stripping Parts

Because we track exactly where a parsed model is inside a message, we can cleanly remove just that portion from
the content and re-sync the other parts to align with the new content. This is helpful for removing context
from a conversation that you might not want there for future generations.

This is a very powerful primitive, that allows you to operate on messages more like a collection of structured
models than raw text.

```py
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

# Gracefully handle missing models
reasoning = meaning.last.try_parse(Reasoning)
if reasoning:
    print("Reasoning:", reasoning.content)

# Strip parsed content to avoid sharing
# previous thoughts with the model.
without_reasons = meaning.strip(Reasoning)
print("Meaning of life:", without_reasons.last.content)

follow_up = without_reasons.continue_(...).run()
```

## Metadata

Both Chats and PendingChats support the concept of arbitrary metadata that you can use to
store things like tags, metrics, and supporting data for storage, sorting, and filtering.

- [`PendingChat.meta()`][rigging.chat.PendingChat.meta] adds to [`PendingChat.metadata`][rigging.chat.PendingChat.metadata]
- [`Chat.meta()`][rigging.chat.Chat.meta] adds to [`Chat.metadata`][rigging.chat.Chat.metadata]

Metadata will carry forward from a PendingChat to a Chat object when generation completes. This
metadata is also maintained in the [serialization process](serialization.md).

```py
import rigging as rg

pending = rg.get_generator("claude-2.1").chat("Hello!").meta(prompt_version=1)
chat = pending.run().meta(user="Will")

print(chat.metadata)
# {
#   'prompt_version': 1, 
#   'user': 'Will'
# }
```