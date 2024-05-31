<h1 align="center">
        🏴‍☠️ Rigging
    </h1>
    <p align="center">
        <img src="riggig-logo.svg" alt="Rigging Logo" width="150" align='center'/>
        <p align="center">The lightweight python LLM interaction framework you've been looking for.
        <br>

Rigging is a lightweight LLM interaction framework built on Pydantic XML. The goal is to make leveraging LLMs in production pipelines as simple and effictive as possible. Here are the highlights:

- **Structured Pydantic models** can be used interchangably with unstructured text output.
- LiteLLM as the default generator giving you **instant access to a huge array of models**.
- Add easy **tool calling** abilities to models which don't natively support it.
- Store different models and configs as **simple connection strings** just like databases.
- Chat templating, forking, continuations, generation parameter overloads, stripping segments, etc.
- Modern python with type hints, async support, pydantic validation, serialization, etc.

```py
import igging as rg
from rigging.model import CommaDelimitedAnswer as Answer

chat = rg.get_generator('gpt-4') \
    .chat(f"Give me 3 famous authors between {Answer.xml_tags()} tags.") \
    .until_parsed_as(Answer) \
    .run()

answer = chat.last.parse(Answer)
print(answer.items)

# ['J. R. R. Tolkien', 'Stephen King', 'George Orwell']
```

Rigging is built and maintained by [dreadnode](https://dreadnode.io) where we use it daily for our work.

## Installation
We publish every version to Pypi:
```bash
pip install rigging
```

If you want to build from source:
```bash
cd rigging/
poetry install
```

## Useage 

### Generators ([**Docs**](https://rigging.dreadnode.io/topics/generators/))

```py
import rigging as rg 

generator = rg.get_generator("claude-3-sonnet-20240229") 
pending = generator.chat(
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
)
chat = pending.run()
print(chat.all)
```

### Chats and Messages [**Docs**](https://rigging.dreadnode.io/topics/chats-and-messages/)

Chat objects hold a sequence of Message objects pre and post generation. This is the most common way that we interact with LLMs, and the interface of both these and PendingChat's are very flexible objects that let you tune the generation process, gather structured outputs, validate parsing, perform text replacements, serialize and deserialize, fork conversations, etc.

```python
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

### Chats to Pandas

Rigging supports various data serialization options for core objects. Chats can be converted to a pandas dataframe as such:

```python
import rigging as rg
from rigging.model import CommaDelimitedAnswer as Answer


chat = rg.get_generator('gpt-4') \
    .chat(f"Give me 3 famous authors between {Answer.xml_tags()} tags.") \
    .until_parsed_as(Answer) \
    .run()

chat.to_df()

```

| chat_id                              | chat_metadata   | chat_generator_id   | chat_timestamp             | generated   | message_id                           | role      | content                                                                                  | parts                                                                                  |
|:-------------------------------------|:----------------|:--------------------|:---------------------------|:------------|:-------------------------------------|:----------|:-----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| 62758800-8797-4832-92ba-bee9ad923ec7 | {}              | litellm!gpt-4       | 2024-05-31 12:31:25.774000 | False       | 1c9f3021-5932-4b55-bb15-f9d182e54a5b | user      | Give me 3 famous authors between <comma-delimited-answer></comma-delimited-answer> tags. | []                                                                                     |
| 62758800-8797-4832-92ba-bee9ad923ec7 | {}              | litellm!gpt-4       | 2024-05-31 12:31:25.774000 | True        | b20da004-d54e-4c25-b287-e41f42bc6888 | assistant | <comma-delimited-answer>J.K. Rowling, Stephen King, Jane Austen</comma-delimited-answer> | [{"model": {"content": "J.K. Rowling, Stephen King, Jane Austen"}, "slice_": [0, 88]}] |


## Getting Started

Head over to **[our documentation](https://rigging.dreadnode.io)** for more information.