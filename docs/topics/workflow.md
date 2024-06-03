## Workflow

1. Get a [`Generator`][rigging.generator.Generator] object - usually with [`get_generator()`][rigging.generator.get_generator].
2. Call [`generator.chat()`][rigging.generator.Generator.chat] to produce a [`ChatPipeline`][rigging.chat.ChatPipeline] and ready it for generation.
3. Call [`pending.run()`][rigging.chat.ChatPipeline.run] to kick off generation and get your final [`Chat`][rigging.chat.Chat] object.

[`ChatPipeline`][rigging.chat.ChatPipeline] objects hold any messages waiting to be delivered to an LLM in exchange
for a new response message. These objects are also where most of the power in rigging comes from. You'll build a
generation pipeline with options, parsing, callbacks, etc. After prep this pending chat is converted into a 
final [`Chat`][rigging.chat.Chat] which holds all messages prior to generation ([`.prev`][rigging.chat.Chat.prev]) 
and after generation ([`.next`][rigging.chat.Chat.next]).

You should think of [`ChatPipeline`][rigging.chat.ChatPipeline] objects like the configurable pre-generation step
with calls like [`.with_()`][rigging.chat.ChatPipeline.with_], [`.apply()`][rigging.chat.ChatPipeline.apply], 
[`.until()`][rigging.chat.ChatPipeline.until], [`.using()`][rigging.chat.ChatPipeline.using], etc. Once you call one
of the many [`.run()`][rigging.chat.ChatPipeline.run] functions, the generator is used to produce the next 
message (or many messages) based on the prior context and any constraints you have in place. Once you have a 
[`Chat`][rigging.chat.Chat] object, the interation is "done" and you can inspect and operate on the messages.

??? tip "Chats vs Completions"

    Rigging supports both Chat objects (messages with roles in a "conversation" format), as well
    as raw text completions. While we use Chat objects in most of our examples, you can check
    out the [Completions](completions.md) section to learn more about their feature parity.

You'll often see us use functional styling chaining as most of our
utility functions return the object back to you.

```go
chat = (
    generator.chat(...)
    .using(...)
    .until(...)
    .with_(...)
    .run()
)
```