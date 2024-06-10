# Callbacks and Mapping

Rigging is designed to give control over how the generation process works, and what occurs after.
In fact, higher level functions like [`.using()`][rigging.chat.ChatPipeline.using]
and [`.until_parsed_as()`][rigging.chat.ChatPipeline] leverage a generic callback system
underneath to guide generation. Let's walk through them.

## Watch Callbacks

Pipelines, Prompts, and Generators hold a list of passive callbacks which will be passed
[`Chat`][rigging.chat.Chat] or [`Completion`][rigging.completion.Completion] objects
as they are generated.

Watch callbacks are useful for logging, monitoring, or other passive actions that don't
directly affect the generation process. Register them with any of the following:

- [`Generator.watch()`][rigging.generator.Generator.watch]
- [`ChatPipeline.watch()`][rigging.chat.ChatPipeline.watch]
- [`CompletionPipeline.watch()`][rigging.completion.CompletionPipeline.watch]
- [`Prompt.watch()`][rigging.prompt.Prompt.watch]

We also provide various helpers in the `rigging.watch` module for writing to
files or databases like elastic.

```py
import rigging as rg

log_to_file = rg.watchers.write_chats_to_jsonl("chats.jsonl")

pipeline = (
    rg.get_generator("gpt-3.5-turbo")
    .chat("Explain why the sky is blue")
    .watch(log_to_file)
)

chat = await pipeline.run_many(5)
```

## Until Callbacks

If you want to gain control over the generation process before it completes,
you can use the [`ChatPipeline.until()`][rigging.chat.ChatPipeline.until] or
[`CompletionPipeline.until()`][rigging.completion.CompletionPipeline.until] methods.

These allow you to register a callback function which participates in generation and
can decide whether generation should proceed, and exactly how it does so. For chat interfaces, these
functions also get fine control over the contents of the chat while callbacks are resolving. This
is how we can provide feedback to an LLM model during generation like validation errors when
parsing fails ([`attempt_recovery`][rigging.chat.ChatPipeline.until]).

```py
import rigging as rg

class Joke(rg.Model):
    content: str

def involves_a_cat(message: rg.Message) -> tuple[bool, list[rg.Message]]:
    if "cat" not in message.content.lower():
        return True, [message, rg.Message("user", "Please include a cat in your joke")] # (1)!
    return False, [message]

chat = (
    await
    rg.get_generator("gpt-3.5-turbo")
    .chat(f"Tell me a joke about an animal between {Joke.xml_tags()} tags.")
    .until_parsed_as(Joke)
    .until(involves_a_cat, drop_dialog=False) # (2)!
    .run()
)

print(chat.conversation)
# [user]: Tell me a joke about an animal between <joke></joke> tags.
# [assistant]: <joke>Why did the duck go to the doctor? Because he was feeling a little down!</joke>
# [user]: Please include a cat in your joke
# [assistant]: <joke>Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!</joke>

print(chat.last.parse(Joke))
# Joke(content='Why was the cat sitting on the computer? Because it wanted to keep an eye on the mouse!')
```

1. Returning `True` from this callback tells Rigging to go back to the generator with the supplied
   messages and rerun the generation step. Whether you're appended messages are used is dependent
   on the `attempt_recovery=True` on [`ChatPipeline.until()`][rigging.chat.ChatPipeline.until]. In
   this instance our request to include a cat will be appending to the intermediate messages while
   generation completes. We can essentially provide feedback to the model about how it should attempt
   to satisfy the callback function.
2. Our use of `drop_dialog=False` here allows us to see the intermediate steps of resolving
   our callbacks in the final Chat. It's up to you whether you want these intermediate messages
   included or not. The default is to drop them once the callbacks resolve.

??? "Using .until on CompletionPipeline"

    The interface for a `CompletionPipeline` is very similar to `ChatPipeline`, except that you
    are only allowed to make a statement about whether generation should retry. You are not
    currently allowed to inject additional text as intermediate context while your callback
    is attempting to resolve.

### Allowing Failures

If you want to allow the generation process to avoid raising an exception when the maximum
rounds is exhausted, you can configure [`on_failed`][rigging.chat.FailMode] on the pipeline, or pass it directly
to various [run methods][rigging.chat.ChatPipeline.run_many] of a `ChatPipeline` or `CompletionPipeline`.

for single runs, pass `allow_failed=True` to [`.run()`][rigging.chat.ChatPipeline.run].

This breaks any guarantees about the validity of final chat objects, but you can check their status
with the [`Chat.failed`][rigging.chat.Chat.failed] or [`Completion.failed`][rigging.completion.Completion.failed] properties.

In the case of `on_failed='skip'`, the final outputs of any run method could be anywhere
from an empty list to a complete list of the requested batch/many.

=== "Allowing Failures"

    ```py hl_lines="11"
    import rigging as rg

    class ValidName(rg.Model):
        ...

    chat = (
        await
        rg.get_generator("gpt-3.5-turbo")
        .chat(f"Provide a fake name between {ValidName.xml_tags()} tags.")
        .until_parsed_as(ValidName)
        .run(allow_failed=True)
    )

    if chat.failed:
        print("Failed to generate a valid name.")
    else:
        print(chat.last.parse(ValidName))
    ```

=== "Including Failures"

    ```py hl_lines="11"
    import rigging as rg

    class ValidName(rg.Model):
        ...

    chats = (
        await
        rg.get_generator("gpt-3.5-turbo")
        .chat(f"Provide a fake name between {ValidName.xml_tags()} tags.")
        .until_parsed_as(ValidName)
        .run_many(3, on_failed="include")
    )

    for chat in chats:
        if chat.failed:
            print("Failed to generate a valid name.")
        else:
            print(chat.last.parse(ValidName))
    ```

=== "Skipping Failures"

    ```py hl_lines="13"
    import rigging as rg

    class ValidName(rg.Model):
        ...

    count = 5

    chats = (
        await
        rg.get_generator("gpt-3.5-turbo")
        .chat(f"Provide a fake name between {ValidName.xml_tags()} tags.")
        .until_parsed_as(ValidName)
        .run_many(count, on_failed="skip")
    )

    successful = len(chats)
    print(f"Generated {successful} valid names out of {count} attempts.")
    ```

### Defining Failures

By default Rigging will catch [`ExhaustedMaxRoundsError`][rigging.error.ExhaustedMaxRoundsError] and
treat those exceptions as a soft failure you can configure with `on_failed`. However, you can also
add different exceptions to a pipeline with [`.catch()`][rigging.chat.ChatPipeline.catch] which will be
caught and treated as soft failures.

For instance, some APIs might raise exceptions if you cross some threshold for content moderation, and
you don't want these exceptions to interupt large scale pipelines.

```py
import litellm
import rigging as rg

pipeline = (
    rg.get_generator("gpt-3.5-turbo")
    .chat("Tell me about great sci-fi books.")
    .catch(litellm.APIError, on_failed="include") # (1)!
)

chats = await pipeline.run_many(3)
```

1. Here we're adding a custom exception to the pipeline that will be caught and treated as a soft failure.
   In the case of litellm raising an APIError, those chats will be marked as failed and included in the
   final output. You can access the raised error with the [`Chat.error`][rigging.chat.Chat.error] property.

## Then Callbacks

You might prefer to have your callbacks execute after generation completes, and operate on
the Chat/Completion objects from there. This is functionally very similar to
[`ChatPipeline.until()`][rigging.chat.ChatPipeline.until] and might be preferred
to expose more of the parsing internals to your code as opposed to the opaque nature
of other callback types. Use the [`ChatPipeline.then()`][rigging.chat.ChatPipeline.then]
to register any number of callbacks before executing [`ChatPipeline.run()`][rigging.chat.ChatPipeline.run].

!!! tip "Branching Chats"

    A common use case for `.then()` is to branch the conversation based on the output of the
    of previous generations. You can continue to chain `.then()` and `.run()` calls to create
    a set of generations that collapse back to the final call when they complete.

=== "Using .then()"

    ```py
    import rigging as rg

    async def check_animal(chat: rg.Chat) -> rg.Chat | None:
        for animal in ["cat", "dog", "cow", "mouse", "elephant", "chicken"]:
            if animal in chat.last.content.lower():
                pipeline = chat.continue_(f"Why did you pick {animal}?")
                return await pipeline.meta(questioned=True).run()

    pipeline = rg.get_generator("gpt-3.5-turbo").chat("Tell me a joke about an animal.")
    pipeline = pipeline.then(check_animal)
    chats = await pipeline.run_many(3)

    for i, chat in enumerate(chats):
        questioned = chat.metadata.get("questioned", False)
        print(f"--- Chat {i+1} (?: {questioned}) ---")
        print(chat.conversation)
        print()
    ```

=== "Output"

    ```
    --- Chat 1 (?: True) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the cat sit on the computer? To keep an eye on the mouse!

    [user]: Why did you pick cat?

    [assistant]: Because they are purr-fect for computer-related jokes!

    --- Chat 2 (?: False) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the duck go to the doctor? Because he was feeling a little "fowl"!

    --- Chat 3 (?: True) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the chicken join a band? Because it had the drumsticks!

    [user]: Why did you pick chicken?

    [assistant]: Because chickens are always up for a good cluck!
    ```

## Map Callbacks

Rigging also allows you to map process a group of Chats all at once. This is particularly
useful for instances of uses of [`.run_many()`][rigging.chat.ChatPipeline.run_many], 
[`.run_batch()`][rigging.chat.ChatPipeline.run_batch], or their async variants.

You also might want to take certain actions depending on the state of a set of Chats
all at once. For instance, attempting re-generation if a certain % of Chats didn't
meet some criteria.

=== "Using .map()"

    ```py
    import rigging as rg

    async def check_animal(chats: list[rg.Chat]) -> list[rg.Chat]:
        return [
            await chat.continue_(f"Why did you pick that animal?").meta(questioned=True).run()
            if any(a in chat.last.content.lower() for a in ["cat", "dog", "cow", "mouse", "elephant", "chicken"])
            else chat
            for chat in chats
        ]

    chats = (
        await
        rg.get_generator("gpt-3.5-turbo")
        .chat("Tell me a joke about an animal.")
        .map(check_animal)
        .run_many(3)
    )

    for i, chat in enumerate(chats):
        questioned = chat.metadata.get("questioned", False)
        print(f"--- Chat {i+1} (?: {questioned}) ---")
        print(chat.conversation)
        print()
    ```

=== "Output"

    ```
    --- Chat 1 (?: True) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the duck cross the road? To prove he wasn't chicken!

    [user]: Why did you pick that animal?

    [assistant]: I chose a duck because they're known for their sense of humor and whimsical nature! Plus, who doesn't love a good duck joke?

    --- Chat 2 (?: True) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the chicken join a band? Because it had the drumsticks!

    [user]: Why did you pick that animal?

    [assistant]: I chose a chicken because they are often associated with funny jokes and puns due to their quirky and comedic behavior. Plus, who doesn't love a good chicken joke?

    --- Chat 3 (?: False) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the duck go to the doctor? Because he was feeling a little down in the dumps!
    ```