# Completions

The majority of Rigging was built around "instruct" or "chat" LLM interfaces where
a base model has been tuned to work with a structured layer on top of raw text completion. We typically
find that base models are more unpredictable with their outputs, tend to be more sensitive to small
changes in their context windows, and require frequent use of [stop tokens][rigging.generator.GenerateParams.stop]
to prevent unneccesary generation.

However, there are some places where completing raw text and working with base models might be desirable:

- Fewer restrictions on the types of content they will generate
- Speeding up generation and lowering token usage by discouraging verbose responses
- Leveraging prompts from popular libraries like [LangChain](https://python.langchain.com/) which assume
  a completions-style interface

## Interface Parity

While we try to maintain parity between the "Chat" and "Completions" interfaces in Rigging, you'll
find some deviations here and there. Completions should be a simple transition if you are familiar
with the other code in rigging. Here are the highlights:

- [`chat`][rigging.generator.Generator.chat] -> [`complete`][rigging.generator.Generator.complete]
- [`Chat`][rigging.chat.Chat] -> [`Completion`][rigging.completion.Completion]
- [`PendingChat`][rigging.chat.PendingChat] -> [`PendingCompletion`][rigging.completion.PendingCompletion]
- [`generate_messages`][rigging.generator.Generator.generate_messages] -> [`generate_texts`][rigging.generator.Generator.generate_texts]

On all of these interfaces, you'll note that sequences of [`Message`][rigging.message.Message] objects have been
replaced with basic `str` objects for both inputs and ouputs.

## Translator Example

Let's build a simply translator object that we can store as a [`PendingCompletion`][rigging.completion.PendingCompletion]
and use it quickly translate a phrase to 3 different languages.

```py
PROMPT = """\
As an expert translator, you accept english text and translate it to $language.

# Format

Input: [english text]
Output: [translated text]
---

Input: $input
Output: """

translator = (
    rg.get_generator('gpt-3.5-turbo') # (1)!
    .complete(PROMPT)
    .with_(stop=["---", "Input:", "\n\n"]) # (2)!
)

text = "Could you please tell me where the nearest train station is?"

for language in ["spanish", "french", "german"]:
    completion = translator.apply(
        language=language,
        input=text
    ).run()
    print(f"[{language}]: {completion.generated}")

# [spanish]: ¿Podría decirme por favor dónde está la estación de tren más cercana?
# [french]:  Pouvez-vous me dire où se trouve la gare la plus proche, s'il vous plaît ?
# [german]:  Könnten Sie mir bitte sagen, wo sich der nächste Bahnhof befindet?
```

1. OpenAPI supports the same model IDs for both completions and chats, but other
   providers might require you to specify a specific model ID used for text completions.
2. We use [`.with_()`][rigging.completion.PendingCompletion.with_] to set stop tokens
   and prevent the generation from simply continuing until our max tokens are reached. This
   is a very common and often required pattern when doing completions over chats. Here, we 
   aren't totally sure what the model might generate after our translation, so
   we use a few different token sequences to be safe.

!!! tip "Using .apply()"

    Text completion is a great place to use the [`.apply`][rigging.completion.PendingCompletion.apply]
    method as we can easily slot in our inputs without using [`.add`][rigging.completion.PendingCompletion.add]
    and following it with our output section of the prompt.