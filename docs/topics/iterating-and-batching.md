# Iterating and Batching

Rigging has good support for iterating over messages, params, and generators, as well as 
large batching of requests. How efficiently these mechanisms operates is dependent on the
underlying generator that's being used, but Rigging has been developed with scale in mind.

## Multiple Generations

The `run_many` functions let you scale out generation N times with the same inputs:

- [`ChatPipeline.run_many()`][rigging.chat.ChatPipeline.run_many]
- [`CompletionPipeline.run_many()`][rigging.completion.CompletionPipeline.run_many]
- [`Prompt.run_many()`][rigging.prompt.Prompt.run_many]

=== "Run Many Code"

    ```py
    import rigging as rg

    async def check_animal(chats: list[rg.Chat]) -> list[rg.Chat]:
        return [
            await chat.continue_(f"Why did you pick that animal?").meta(questioned=True).run()
            if any(a in chat.last.content.lower() for a in ["cat", "dog", "cow", "mouse"])
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

The `run_batch` functions let you batch accross a set of inputs:

- [`ChatPipeline.run_batch()`][rigging.chat.ChatPipeline.run_batch]
- [`CompletionPipeline.run_batch()`][rigging.completion.CompletionPipeline.run_batch]
 
As processing proceeds with things like [`.then`][rigging.chat.ChatPipeline.then] or
[`.map`][rigging.chat.ChatPipeline.map], that chats will resolve individually and
collapse into the final results.

=== "Batching Inputs Code"

    ```py
    import rigging as rg
    from rigging.model import CommaDelimitedAnswer

    pipeline = (
        rg.get_generator('gpt-4-turbo')
        .chat({
            "role": "system",
            "content": f"Always respond with {CommaDelimitedAnswer.xml_tags()} tags."}
        )
        .until_parsed_as(CommaDelimitedAnswer, attempt_recovery=True)
    )

    many = [f"Give me 3 famous {thing}" for thing in ["authors", "painters", "musicians", "hackers"]]

    chats = await pipeline.run_batch(many, on_failed='skip')

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
    ['Michael Jackson', 'Beyonce', 'The Beatles']
    ```

!!! tip "Skipping failed results"

    Passing `on_failed='skip'` to [`.run_batch`][rigging.chat.ChatPipeline.run_batch], or configuring
    a pipeline with [`.catch(..., on_failed='skip')`][rigging.chat.ChatPipeline.catch] will cause the function to
    ignore any parsing errors like [`ExhaustedMaxRoundsError`][rigging.error.ExhaustedMaxRoundsError] and only
    return the chats that were successful.


## Batching Parameters

In addition to batching against input messages or strings, you can fix a single input
and build a batch accross a set of generation parameters. The inputs to
[`.run_batch`][rigging.chat.ChatPipeline.run_batch]
will scale either the generate parameters or the input messages if either is a single item.

=== "Batching Code"

    ```py
    import rigging as rg

    pipeline = rg.get_generator("gpt-3.5-turbo").chat()

    chats = await pipeline.run_batch(
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

## Iterating over Models

The `run_over` functions let you execute generation over a set of generators:

- [`ChatPipeline.run_over()`][rigging.chat.ChatPipeline.run_over]
- [`CompletionPipeline.run_over()`][rigging.completion.CompletionPipeline.run_over]
- [`Prompt.run_over()`][rigging.prompt.Prompt.run_over]

Generators can be passed as string identifiers or full instances of 
[Generator][rigging.generator.Generator].

By default the original generator associated with the [`ChatPipeline`][rigging.chat.ChatPipeline]
is included in the iteration, configurable with the `include_original` parameter.

Much like the [`run_many`][rigging.chat.ChatPipeline.run_many]
and [`run_batch`][rigging.chat.ChatPipeline.run_batch] functions, you can control the
handling of failures with the `on_failed` parameter.

=== "Run Over Code"

    ```py
    import rigging as rg
    from rigging.model import Answer

    QUESTION = "What is the capital of France?"
    ANSWER = "paris"

    async def score_output(chats: list[rg.Chat]) -> list[rg.Chat]:
        return [
            chat.meta(correct=chat.last.parse(Answer).content.lower() == ANSWER)
            for chat in chats
        ]

    chats = (
        await
        rg.get_generator("gpt-3.5-turbo")
        .chat([
            {"role": "system", "content": f"Always respond in one word between {Answer.xml_tags()} tags."},
            {"role": "user", "content": QUESTION}
        ])
        .until_parsed_as(Answer, max_rounds=3)
        .map(score_output)
        .run_over("gpt-4-turbo", "claude-3-haiku-20240307,temperature=0.5", "claude-3-sonnet-20240229")
    )

    for chat in chats:
        print("Model: ", chat.generator.model)
        print("Msg:   ", chat.last.content)
        print("Meta:  ", chat.metadata)
        print()
    ```

=== "Outputs"

    ```
    Model: gpt-4-turbo
    Msg:   <answer>Paris</answer>
    Meta:  {'correct': True}

    Model: claude-3-haiku-20240307
    Msg:   <answer>Paris</answer>
    Meta:  {'correct': True}

    Model: claude-3-sonnet-20240229
    Msg:   <answer>Paris</answer>
    Meta:  {'correct': True}

    Model: openai/gpt-3.5-turbo
    Msg:   <answer>Paris</answer>
    Meta:  {'correct': True}
    ```