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
import rigging as rg
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

### API Keys

All generators carry a .api_key attribute which can be set directly, or by passing ,api_key= as part of an identifier string. Not all generators will require one, but they are common enough that we include the attribute as part of the base class.

Typically you will be using a library like LiteLLM underneath, and can simply use environment variables:

```bash
export OPENAI_API_KEY=...
export TOGETHER_API_KEY=...
export TOGETHERAI_API_KEY=...
export MISTRAL_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Supported Models

Rigging supports various models out of the box using the LiteLMM or local models using a vLLM server. Models include

### Via LiteLLM

Includes but not limited to:
- openai
- azure
- aws
- google[gemini]
- mistral
- anthorpic

Full list via the liteLLM docs [here](https://github.com/BerriAI/litellm/blob/main/README.md#supported-providers-docs)


### Local Models

Rigging also supports local models hosted via vLLM. This allows for seamless integration with popular Hugging Face models lie:
Includes but not limited to:

- llama3
- phi3
- mixtral
- gemma

Example for connecting to a locally hosted llama3 model:

```python
g_vllm_port = 9999
g_vllm_model_name = "llama3"

generator = rg.get_generator(
    f"openai/{g_vllm_model_name}," \
    f"api_base=http://localhost:{g_vllm_port}/v1," \
    "api_key=sk-1234," \
    "temperature=0.9,max_tokens=512," \
    "stop=<|eot_id|>" # Llama requires some hand holding,
)
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

### Generater Parameters ([**Docs**](https://rigging.dreadnode.io/api/generator/#rigging.generator.GenerateParams))

We can set model parameters using the `rg.GenerateParams` class. This class allows you to set various model parameters including:
```
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    api_base: str | None = None,
    timeout: int | None = None,
    seed: int | None = None,
    extra: dict[str, typing.Any] = None,
```

Example of calling a generator chat with cusom model parameters might look like:

```python
rg_params = rg.GenerateParams(
    temperature = 0.9,
    max_tokens = 512,
)
base_chat = generator.chat(params=rg_params)
answer = base_chat.fork('How is it going?').run()
print(answer.last.content)
```

### Data Models ([**Docs**](https://rigging.dreadnode.io/topics/models/))

Model definitions are at the core of Rigging, and provide an extremely powerful interface of defining exactly what kinds of input data you support and how it should be validated.

```python
from pydantic import StringConstraints

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]
str_upper = t.Annotated[str, StringConstraints(to_upper=True)]

class Header(rg.Model):
    name: str = rg.attr()
    value: str_strip

class Parameter(rg.Model):
    name: str = rg.attr()
    value: str_strip

class Request(rg.Model):
    method: str_upper = rg.attr()
    path: str = rg.attr()
    headers: list[Header] = rg.wrapped("headers", rg.element(default=[]))
    url_params: list[Parameter] = rg.wrapped("url-params", rg.element(default=[]))
    body: str_strip = rg.element(default="")
```


### Chats and Messages ([**Docs**](https://rigging.dreadnode.io/topics/chats-and-messages/))

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

### Data Serialization to Pandas DataFrame ([**Docs**](https://rigging.dreadnode.io/topics/serialization/))

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

Will output:

| chat_id                              | chat_metadata   | chat_generator_id   | chat_timestamp             | generated   | message_id                           | role      | content                                                                                  | parts                                                                                  |
|:-------------------------------------|:----------------|:--------------------|:---------------------------|:------------|:-------------------------------------|:----------|:-----------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------|
| 62758800-8797-4832-92ba-bee9ad923ec7 | {}              | litellm!gpt-4       | 2024-05-31 12:31:25.774000 | False       | 1c9f3021-5932-4b55-bb15-f9d182e54a5b | user      | Give me 3 famous authors between <comma-delimited-answer></comma-delimited-answer> tags. | []                                                                                     |
| 62758800-8797-4832-92ba-bee9ad923ec7 | {}              | litellm!gpt-4       | 2024-05-31 12:31:25.774000 | True        | b20da004-d54e-4c25-b287-e41f42bc6888 | assistant | <comma-delimited-answer>J.K. Rowling, Stephen King, Jane Austen</comma-delimited-answer> | [{"model": {"content": "J.K. Rowling, Stephen King, Jane Austen"}, "slice_": [0, 88]}] |


### Async and Batching ([**Docs**](https://rigging.dreadnode.io/topics/async-and-batching/))

Rigging has good support for handling async generation and large batching of requests. How efficiently these mechanisms operates is dependent on the underlying generator that's being used, but Rigging has been developed with scale in mind.

The `.run_many` and `.arun_many` functions let you take the same inputs and generation parameters, and simply run the generation multiple times.

```python
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

# Output:
#
# --- Chat 1 (?: False) ---
# [user]: Tell me a joke about an animal.

# [assistant]: Why did the spider go to the computer? 

# To check his website!

# --- Chat 2 (?: False) ---
# [user]: Tell me a joke about an animal.

# [assistant]: Why did the chicken join a band? Because it had the drumsticks!

# --- Chat 3 (?: True) ---
# [user]: Tell me a joke about an animal.

# [assistant]: Why don't elephants use computers?

# Because they're afraid of the mouse!

# [user]: Why did you pick that animal?

# [assistant]: I chose an elephant because they are known for their intelligence and gentle nature, making them a popular subject for jokes and humorous anecdotes. Plus, imagining an elephant trying to use a computer and being scared of a tiny mouse is a funny visual image!
```

## Support and Discuss with our Founders

This project is built and supported by dreadnode. Sign up for our email list or schedule a call through our website: https://www.dreadnode.io/

## Documentation

Head over to **[our documentation](https://rigging.dreadnode.io)** for more information.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dreadnode/rigging&type=Date)](https://star-history.com/#dreadnode/rigging&Date)