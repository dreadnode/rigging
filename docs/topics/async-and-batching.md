# Async and Batching

Rigging has good support for handling async generation and large batching of requests. How efficiently
these mechanisms operates is dependent on the underlying generator that's being used, but Rigging has
been developed with scale in mind.

## Multiple Generations

The [`.run_many`][rigging.chat.ChatPipeline.run_many] and [`.arun_many`][rigging.chat.ChatPipeline.arun_many] functions
let you take the same inputs and generation parameters, and simply run the generation multiple times.

=== "Run Many Code"

    ```py
    import rigging as rg

    def check_animal(chats: list[rg.Chat]) -> list[rg.Chat]:
        return [
            chat.continue_(f"Why did you pick that animal?").meta(questioned=True).run()
            if any(a in chat.last.content.lower() for a in ["cat", "dog", "cow", "mouse"])
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
    --- Chat 1 (?: False) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the spider go to the computer? 

    To check his website!

    --- Chat 2 (?: False) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why did the chicken join a band? Because it had the drumsticks!

    --- Chat 3 (?: True) ---
    [user]: Tell me a joke about an animal.

    [assistant]: Why don't elephants use computers?

    Because they're afraid of the mouse!

    [user]: Why did you pick that animal?

    [assistant]: I chose an elephant because they are known for their intelligence and gentle nature, making them a popular subject for jokes and humorous anecdotes. Plus, imagining an elephant trying to use a computer and being scared of a tiny mouse is a funny visual image!
    ```

## Batching Inputs

You can use the [`.run_batch`][rigging.chat.ChatPipeline.run_batch] and [`.arun_batch`][rigging.chat.ChatPipeline.arun_batch]
functions to batch accross a set of inputs and collect all the chats. As processing proceeds with things like
[`.then`][rigging.chat.ChatPipeline.then] or [`.until_parsed_as`][rigging.chat.ChatPipeline.until_parsed_as], that chats
will resolve individually and collapse into the final results.

=== "Batching Inputs Code"

    ```py
    import rigging as rg
    from rigging.model import CommaDelimitedAnswer

    pending = (
        rg.get_generator('gpt-3.5-turbo')
        .chat({
            "role": "system",
            "content": f"Always respond with {CommaDelimitedAnswer.xml_tags()} tags."}
        )
        .until_parsed_as(CommaDelimitedAnswer, attempt_recovery=True)
    )

    many = [f"Give me 3 famous {thing}" for thing in ["authors", "painters", "musicians", "hackers"]]

    chats = await pending.arun_batch(many, skip_failed=True)

    for i, chat in enumerate(chats):
        print(f"--- Chat {i+1} ({len(chat)}) ---")
        print(chat.last.parse(CommaDelimitedAnswer).items)
        print()
    ```

=== "Outputs"

    ```
    --- Chat 1 (2) ---
    ['Leonardo da Vinci', 'Vincent van Gogh', 'Pablo Picasso']

    --- Chat 2 (2) ---
    ['Michael Jackson', 'Beyonc&#233;', 'The Beatles']
    ```

!!! tip "Skipping failed results"

    Passing `skip_failed=True` to [`.run_batch`][rigging.chat.ChatPipeline.run_batch] will cause the function to
    ignore any parsing errors like [`ExhaustedMaxRoundsError`][rigging.error.ExhaustedMaxRoundsError] and only
    return the chats that were successful.


## Batching Parameters

In addition to batching against input messages or strings, you can fix a single input
and build a batch accross a set of generation parameters. The inputs to
[`.run_batch`][rigging.chat.ChatPipeline.run_batch] and [`.arun_batch`][rigging.chat.ChatPipeline.arun_batch]
will scale either the generate parameters or the input messages if either is a single item.

=== "Batching Code"

    ```py
    import rigging as rg

    pending = rg.get_generator("gpt-3.5-turbo").chat()

    chats = await pending.arun_batch(
        ["Tell me a short fact about an japanese city."],
        [rg.GenerateParams(temperature=t) for t in [0.6, 0.9, 1.2, 1.5, 1.8]]
    )

    for i, chat in enumerate(chats):
        print(f"--- Chat {i+1} ---")
        print(chat.generator_id)
        print()
        print(chat.conversation)
        print()
    ```

=== "Outputs"

    ```
    --- Chat 1 ---
    litellm!gpt-3.5-turbo,temperature=0.6

    [assistant]: Tokyo, the capital city of Japan, is the most populous
    metropolitan area in the world, with over 37 million residents.

    --- Chat 2 ---
    litellm!gpt-3.5-turbo,temperature=0.9

    [assistant]: Tokyo is the largest metropolitan area in the world,
    with a population of over 37 million people.

    --- Chat 3 ---
    litellm!gpt-3.5-turbo,temperature=1.2

    [assistant]: Kyoto, a city in Japan known for its historic temples
    and gardens, was once the capital of Japan for over 1,000 years from
    794 until the capital was moved to Tokyo in 1869.

    --- Chat 4 ---
    litellm!gpt-3.5-turbo,temperature=1.5

    [assistant]: Nagoya, Japan is known for being one of the leading
    manufacturing and industrial regions in the country, with a strong
    automotive presence including major factories for Toyota, Honda, and Mitsubishi.

    --- Chat 5 ---
    litellm!gpt-3.5-turbo,temperature=1.8

    [assistant]: Sendai is the largest city in the Tohoku region of
    Japan and is known for its incredible natural scenery, such as the
    nearby Sendai Bay and Zuihoden mausoleum.
    ```