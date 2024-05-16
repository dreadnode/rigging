# Serialization

The following objects in Rigging have great serialization support for storage and retrieval:

- [`Chat`][rigging.chat.Chat]
- [`Completion`][rigging.completion.Completion]
- [`Generator`][rigging.generator.Generator]
- [`Model`][rigging.model.Model]

Most of this stems from our use of Pydantic for core models, and we've included some helpful
fields for reconstructing Chats and Completions.

## JSON Serialization

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

## JSON Deserialization

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

## Pandas DataFrames

Rigging also has helpers in the [`rigging.data`][] module for performing conversions
between Chat objects and other storage formats like Pandas. In [`chats_to_df`][rigging.data.chats_to_df]
the messages are flattened and stored with a `chat_id` column for grouping.
[`df_to_chats`][rigging.data.df_to_chats] allows you to reconstruct a list of Chat objects back from a DataFrame.

```py
import rigging as rg

chats = (
    rg.get_generator("claude-3-haiku-20240307")
    .chat("Write me a haiku.")
    .run_many(3)
)

df = rg.chats_to_df(chats)

print(df.info())

# RangeIndex: 6 entries, 0 to 5
# Data columns (total 9 columns):
#  #   Column             Non-Null Count  Dtype         
# ---  ------             --------------  -----         
#  0   chat_id            6 non-null      string        
#  1   chat_metadata      6 non-null      string        
#  2   chat_generator_id  6 non-null      string        
#  3   chat_timestamp     6 non-null      datetime64[ms]
#  4   generated          6 non-null      bool          
#  5   role               6 non-null      category      
#  6   parts              6 non-null      string        
#  7   content            6 non-null      string        
#  8   message_id         6 non-null      string        
# dtypes: bool(1), category(1), datetime64[ms](1), string(6)

df.content.apply(lambda x: len(x)).mean()

# 60.166666666666664

back = rg.df_to_chats(df)
print(back[0].conversation)

# [user]: Write me a haiku.
# 
# [assistant]: Here's a haiku for you:
# 
# Gentle breeze whispers,
# Flowers bloom in vibrant hues,
# Nature's simple bliss.
```