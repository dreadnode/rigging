# Callbacks and Mapping

Rigging is designed to give control over how the generation process works, and what occurs after. In fact, 
higher level functions like [`.using()`][rigging.chat.PendingChat.using] and [`.until_parsed_as()`][rigging.chat.PendingChat]
leverage a generic callback system underneath to guide generation. Let's walk through them.

## Until Callbacks

If you want to gain control over the generation process before it completes, you can use the
[`PendingChat.until()`][rigging.chat.PendingChat.until] or [`PendingCompletion.until()`][rigging.completion.PendingCompletion.until]
methods. These allow you to register a callback function which participates in generation and
can decide whether generation should proceed, and exactly how it does so. For chat interfaces, these
functions also get fine control over the contents of the chat while callbacks are resolving. This
is how we can provide feedback to an LLM model during generation like validation errors when
parsing fails ([`attempt_recovery`][rigging.chat.PendingChat.until]).

```py
import rigging as rg

class Joke(rg.Model):
    content: str

def involves_a_cat(message: rg.Message) -> tuple[bool, list[rg.Message]]:
    if "cat" not in message.content.lower():
        return True, [message, rg.Message("user", "Please include a cat in your joke")] # (1)!
    return False, [message]

chat = (
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
   on the `attempt_recovery=True` on [`PendingChat.until()`][rigging.chat.PendingChat.until]. In
   this instance our request to include a cat will be appending to the intermediate messages while
   generation completes. We can essentially provide feedback to the model about how it should attempt
   to satisfy the callback function.
2. Our use of `drop_dialog=False` here allows us to see the intermediate steps of resolving
   our callbacks in the final Chat. It's up to you whether you want these intermediate messages
   included or not. The default is to drop them once the callbacks resolve.

??? "Using .until on PendingCompletion"

    The interface for a `PendingCompletion` is very similar to `PendingChat`, except that you
    are only allowed to make a statement about whether generation should retry. You are not
    currently allowed to inject additional text as intermediate context while your callback
    is attempting to resolve.

### Allowing Failures

If you want to allow the generation process to avoid raising an exception when the maximum
rounds is exhausted, you can pass `allow_failed=True`, `include_failed=True`, or `skip_failed=True`
to the various [run methods][rigging.chat.PendingChat.run] of a `PendingChat` or `PendingCompletion`.

This breaks any guarantees about the validity of final chat objects, but you can check their status
with the [`Chat.failed`][rigging.chat.Chat.failed] or [`Completion.failed`][rigging.completion.Completion.failed] properties.

In the case of `skip_failed`, the final outputs of any run method could be anywhere from an empty list
to a complete list of the requested batch/many.

=== "Allowing Failures"

    ```py hl_lines="10"
    import rigging as rg

    class ValidName(rg.Model):
        ...

    chat = (
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

    ```py hl_lines="10"
    import rigging as rg

    class ValidName(rg.Model):
        ...

    chats = (
        rg.get_generator("gpt-3.5-turbo")
        .chat(f"Provide a fake name between {ValidName.xml_tags()} tags.")
        .until_parsed_as(ValidName)
        .run_many(3, include_failed=True)
    )

    for chat in chats:
        if chat.failed:
            print("Failed to generate a valid name.")
        else:
            print(chat.last.parse(ValidName))
    ```

=== "Skipping Failures"

    ```py hl_lines="12"
    import rigging as rg

    class ValidName(rg.Model):
        ...

    count = 5

    chats = (
        rg.get_generator("gpt-3.5-turbo")
        .chat(f"Provide a fake name between {ValidName.xml_tags()} tags.")
        .until_parsed_as(ValidName)
        .run_many(count, skip_failed=True)
    )

    successful = len(chats)
    print(f"Generated {successful} valid names out of {count} attempts.")
    ```

## Then Callbacks

You might prefer to have your callbacks execute after generation completes, and operate on
the Chat/Completion objects from there. This is functionally very similar to [`PendingChat.until()`][rigging.chat.PendingChat.until]
and might be preferred to expose more of the parsing internals to your code as opposed to
the opaque nature of other callback types. Use the [`PendingChat.then()`][rigging.chat.PendingChat.then]
to register any number of callbacks before executing [`PendingChat.run()`][rigging.chat.PendingChat.run].

!!! tip "Branching Chats"

    A common use case for `.then()` is to branch the conversation based on the output of the
    of previous generations. You can continue to chain `.then()` and `.run()` calls to create
    a set of generations that collapse back to the final call when they complete.

??? tip "Async Callbacks"

    You are free to define async versions of your callbacks here, but the type of callbacks
    registered has to match your use of either sync [`.run()`][rigging.chat.PendingChat.run] variants
    or their async [`.arun()`][rigging.chat.PendingChat.arun] versions.

=== "Using .then()"

    ```py
    import rigging as rg

    def check_animal(chat: rg.Chat) -> rg.Chat | None:
        for animal in ["cat", "dog", "cow", "mouse", "elephant", "chicken"]:
            if animal in chat.last.content.lower():
                pending = chat.continue_(f"Why did you pick {animal}?")
                return pending.meta(questioned=True).run()

    pending = rg.get_generator("gpt-3.5-turbo").chat("Tell me a joke about an animal.")
    pending = pending.then(check_animal)
    chats = pending.run_many(3)

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
useful for instances of uses of [`.run_many()`][rigging.chat.PendingChat.run_many], 
[`.run_batch()`][rigging.chat.PendingChat.run_batch], or their async variants.

You also might want to take certain actions depending on the state of a set of Chats
all at once. For instance, attempting re-generation if a certain % of Chats didn't
meet some criteria.

??? tip "Async Callbacks"

    You are free to define async versions of your callbacks here, but the type of callbacks
    registered has to match your use of either sync [`.run_many()`][rigging.chat.PendingChat.run_many] variants
    or their async [`.arun_many()`][rigging.chat.PendingChat.run_many] versions.

=== "Using .map()"

    ```py
    import rigging as rg

    def check_animal(chats: list[rg.Chat]) -> list[rg.Chat]:
        return [
            chat.continue_(f"Why did you pick that animal?").meta(questioned=True).run()
            if any(a in chat.last.content.lower() for a in ["cat", "dog", "cow", "mouse", "elephant", "chicken"])
            else chat
            for chat in chats
        ]

    chats = (
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