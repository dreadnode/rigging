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
with calls like [`.with_()`][rigging.chat.PendingChat.with_], [`.apply()`][rigging.chat.PendingChat.apply], 
[`.until()`][rigging.chat.PendingChat.until], [`.using()`][rigging.chat.PendingChat.using], etc. Once you call one
of the many [`.run()`][rigging.chat.PendingChat.run] functions, the generator is used to produce the next 
message (or many messages) based on the prior context and any constraints you have in place. Once you have a 
[`Chat`][rigging.chat.Chat] object, the interation is "done" and you can inspect and operate on the messages.

You'll often see us use functional styling chaining as most of our
utility functions return the object back to you.

```py
chat = generator.chat(...) \
    .using(...).until(...).with_(...) \
    .run()
```