# Serialization

The following objects in Rigging have great serialization support for storage and retrieval:

- [`Chat`][rigging.chat.Chat]
- [`Completion`][rigging.completion.Completion]
- [`Generator`][rigging.generator.Generator]
- [`Model`][rigging.model.Model]

Most of this stems from our use of Pydantic for core models, and we've included some helpful
fields for reconstructing Chats and Completions.

## Serializing Chats

Let's build a joke pipeline and serialize the final chat into JSON.

=== "Serialization Code"

    ```py
    import rigging as rg

    class Joke(rg.Model):
        content: str

    chat = rg.get_generator("gpt-3.5-turbo") \
        .chat(f"Provide 3 jokes each between {Joke.xml_tags()} tags.") \
        .meta(tags=['joke']) \
        .with_(temperature=1.25) \
        .run()

    chat.last.parse_set(Joke)

    serialized = chat.model_dump_json(indent=2)
    print(serialized)
    ```

=== "Serialized JSON"

    ```json
    {
    "uuid": "891c3834-2588-4652-8371-e9746086fd46",
    "timestamp": "2024-05-10T11:44:15.501326",
    "messages": [
        {
        "role": "user",
        "parts": [],
        "content": "Provide 3 jokes each between <joke></joke> tags."
        }
    ],
    "generated": [
        {
        "role": "assistant",
        "parts": [
            {
            "model": {
                "content": " Why was the math book sad? Because it had too many problems. "
            },
            "slice_": [
                0,
                75
            ]
            },
            {
            "model": {
                "content": " I told my wife she should embrace her mistakes. She gave me a hug. "
            },
            "slice_": [
                76,
                157
            ]
            },
            {
            "model": {
                "content": " Why did the scarecrow win an award? Because he was outstanding in his field. "
            },
            "slice_": [
                158,
                249
            ]
            }
        ],
        "content": "<joke> Why was the math book sad? Because it had too many problems. </joke>\n<joke> I told my wife she should embrace her mistakes. She gave me a hug. </joke>\n<joke> Why did the scarecrow win an award? Because he was outstanding in his field. </joke>"
        }
    ],
    "metadata": {
        "tags": [
        "joke"
        ]
    },
    "generator_id": "litellm!gpt-3.5-turbo,temperature=1.25"
    }
    ```

You'll notice that every Chat gets a unique `id` field to help track them in a datastore like Elastic or Pandas. We also
assign a `timestamp` to understand when the generation took place. We are also taking advantage of the
[`.meta()`][rigging.chat.PendingChat.meta] to add a tracking tag for filtering later.

## Deserializing Chats

The JSON has everything required to reconstruct a Chat including a `generator_id` dynamically
constructed to perserve the parameters used to create the generated message(s). We can now
deserialize a chat from a datastore, and immediately step back into a 
[`PendingChat`][rigging.chat.PendingChat] for exploration.

```py
chat = rg.Chat.model_validate_json(serialized)
print(chat.conversation)
# [user]: Provide 3 jokes each between <joke></joke> tags.

# [assistant]: 
# <joke> Why was the math book sad? Because it had too many problems. </joke>
# <joke> I told my wife she should embrace her mistakes. She gave me a hug. </joke>
# <joke> Why did the scarecrow win an award? Because he was outstanding in his field. </joke>

continued = chat.continue_("Please explain the first joke to me.").run()
print(continued.last)
# [assistant]: In the first joke, the pun is based on the double meaning of the word "problems."
# The math book is described as being sad because it has "too many problems," which could be
# interpreted as having both mathematical problems (equations to solve) and emotional difficulties.
# This play on words adds humor to the joke.
```