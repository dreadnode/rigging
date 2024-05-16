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

## Getting Started

Rigging is a flexible library built on top of other very flexible libraries. As such it might take a bit to warm
up to it's interfaces provided the many ways you can accomplish your goals. However, the code is well documented
and topic pages and source are a great places to step in/out of as you explore.

??? tip "IDE Setup"

    Rigging has been built with full type support which provides clear guidance on what
    methods return what types, and when they return those types. It's recommended that you
    operate in a development environment which can take advantage of this information.
    You're use of Rigging will almost "fall" into place and you won't be guessing about
    objects as you work.

### Basic Chats

Let's start with a very basic generation example that doesn't include any parsing features, continuations, etc.
You want to chat with a model and collect it's response.

We first need to get a [generator][rigging.generator.Generator] object. We'll use 
[`get_generator`][rigging.generator.get_generator] which will resolve an identifier string 
to the underlying generator class object.

??? note "API Keys"

    The default Rigging generator is [LiteLLM][rigging.generator.LiteLLMGenerator], which
    wraps a large number of providers and models. We assume for these examples that you
    have API tokens set as environment variables for these models. You can refer to the
    [LiteLLM docs](https://docs.litellm.ai/docs/) for supported providers and their key format.
    If you'd like, you can change any of the model IDs we use and/or add `,api_key=[sk-1234]` to the
    end of any of the generator IDs to specify them inline.

```py hl_lines="3"
import rigging as rg # (1)!

generator = rg.get_generator("claude-3-sonnet-20240229") # (2)!
pending = generator.chat(
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
)
chat = pending.run()
print(chat.all)
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
# ]
```

1. You'll see us use this shorthand import syntax throughout our code, it's
   totally optional but makes things look nice.
2. This is actually shorthand for `litellm!anthropic/claude-3-sonnet-20240229`, where `litellm`
   is the provider. We just default to that generator and you don't have to be explicit. You
   can find more information about this in the [generators](topics/generators.md) docs.


Generators have an easy [`chat()`][rigging.generator.Generator.chat] method which you'll
use to initiate the conversations. You can supply messages in many different forms from
dictionary objects, full [`Message`][rigging.message.Message] classes, or a simple `str`
which will be converted to a user message.

```py hl_lines="4-9"
import rigging as rg

generator = rg.get_generator("claude-3-sonnet-20240229")
pending = generator.chat( # (1)!
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
)
chat = pending.run()
print(chat.all)
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
#   Message(role='assistant', parts=[], content='Hello! How can I help you today?'),
# ]
```

1. [`generator.chat`][rigging.generator.Generator.chat] is actually just a helper for
   [`chat(generator, ...)`][rigging.generator.chat], they do the same thing.

??? note "PendingChat vs Chat"

    You'll notice we name the result of `chat()` as `pending`. The naming might be confusing,
    but chats go through 2 phases. We first stage them into a pending state, where we operate
    and prepare them in a "pipeline" of sorts before we actually trigger generation with `run()`.

    Calling `.chat()` doesn't trigger any generation, but calling any of these run methods will:

    - [rigging.chat.PendingChat.run][]
    - [rigging.chat.PendingChat.run_many][]
    - [rigging.chat.PendingChat.run_batch][]

In this case, we have nothing additional we want to add to our pending chat, and we are only interested
in generating exactly one response message. We simply call [`.run()`][rigging.chat.PendingChat.chat] to
execute the generation process and collect our final [`Chat`][rigging.chat.Chat] object.

```py hl_lines="10-11"
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
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
#   Message(role='assistant', parts=[], content='Hello! How can I help you today?'),
# ]
```

View more about Chat objects and their properties [over here.][rigging.chat.Chat]. In general, chats
give you access to exactly what messages were passed into a model, and what came out the other side.

### Conversation

Both `PendingChat` and `Chat` objects provide freedom for forking off the current state of messages, or
continuing a stream of messages after generation has occured. In general:

- [`PendingChat.fork`][rigging.chat.PendingChat.fork] will clone the current pending chat and let you maintain
  both the new and original object for continued processing.
- [`Chat.fork`][rigging.chat.Chat.fork] will produce a fresh `PendingChat` from all the messages prior to the
  previous generation (useful for "going back" in time).
- [`Chat.continue_`][rigging.chat.Chat.continue_] is similar to `fork` (actually a wrapper) which tells `fork` to
  include the generated messages as you move on (useful for "going forward" in time).
 
```py
import rigging as rg

generator = rg.get_generator("gpt-3.5-turbo")
chat = generator.chat([
        {"role": "user", "content": "Hello, how are you?"},
])

# We can fork before generation has occured
specific = chat.fork("Be specific please.").run()
poetic = chat.fork("Be as poetic as possible").with_(temperature=1.5).run() # (1)!

# We can also continue after generation
next_chat = poetic.continue_(
    {"role": "user", "content": "That's good, tell me a joke"}
)

update = next_chat.run()
```

1. In this case the temperature change will only be applied to the poetic path because `fork` has
   created a clone of our pending chat. 

### Basic Parsing

Now let's assume we want to ask the model for a piece of information, and we want to make sure
this item conforms to a pre-defined structure. Underneath rigging uses [Pydantic XML](https://pydantic-xml.readthedocs.io/)
which itself is built on [Pydantic](https://docs.pydantic.dev/). We'll cover more about
constructing models in a [later section](topics/models.md), but don't stress the details for now.

??? note "XML vs JSON"

    Rigging is opinionated with regard to using XML to weave unstructured data with structured contents
    as the underlying LLM generates text responses. A frequent solution to getting "predictable"
    outputs from LLMs has been forcing JSON conformant outputs, but we think this is
    poor form in the long run. You can read more about this from [Anthropic](https://docs.anthropic.com/claude/docs/use-xml-tags)
    who have done extensive research with their models.

    We'll skip the long rant, but trust us that XML is a very useful syntax which beats
    JSON any day of the week for typical use cases.

To begin, let's define a `FunFact` model which we'll have the LLM fill in. Rigging exposes a 
[`Model`][rigging.model.Model] base class which you should inherit from when defining structured
inputs. This is a lightweight wrapper around pydantic-xml's [`BaseXMLModel`](`https://pydantic-xml.readthedocs.io/en/latest/pages/api.html#pydantic_xml.BaseXmlModel`)
with some added features and functionality to make it easy for Rigging to manage. However, everything
these models support (for the most part) is also supported in Rigging.

```py hl_lines="3-4"
import rigging as rg

class FunFact(rg.Model):
    fact: str # (1)!

chat = rg.get_generator('gpt-3.5-turbo').chat(
    f"Provide a fun fact between {FunFact.xml_example()} tags."
).run()

fun_fact = chat.last.parse(FunFact)

print(fun_fact.fact)
# The Eiffel Tower can be 15 cm taller during the summer due to the expansion of the iron in the heat. 
```

1. This is what pydantic XML refers to as a "primitive" class as it is simply and single
   typed value placed between the tags. See more about primitive types, elements, and attributes in the 
   [Pydantic XML Docs](https://pydantic-xml.readthedocs.io/en/latest/pages/quickstart.html#primitives)

We need to show the target LLM how to format it's response, so we'll use the 
[`.xml_example()`][rigging.model.Model.xml_example] class method which all models
support. By default this will simple emit empty XML tags of our model:

```xml
Provide a fun fact between <fun-fact></fun-fact> tags.
```

??? note "Customizing Model Tags"

    Tags for a model are auto-generated based on the name of the class. You are free
    to override these by passing `tag=[value]` into your class definition like this:

    ```py
    class LongNameForThing(rg.Model, tag="short"):
        ...
    ```

We wrap up the generation and extract our parsed object by calling [`.parse()`][rigging.message.Message.parse]
on the [last message][rigging.chat.Chat.last] of our generated chat. This will process the contents
of the message, extract the first matching model which parses successfully, and return it to us as a python
object.

```py hl_lines="10"
import rigging as rg

class FunFact(rg.Model):
    fact: str

chat = rg.get_generator('gpt-3.5-turbo').chat(
    f"Provide a fun fact between {FunFact.xml_example()} tags."
).run()

fun_fact = chat.last.parse(FunFact)

print(fun_fact.fact) # (1)!
# The Eiffel Tower can be 15 cm taller during the summer due to the expansion of the iron in the heat. 
```

1. Because we've defined `FunFact` as a class, the result if `.parse()` is typed to that object. In our
   code, all the properties of fact will be available just like we created the object directly.

Notice that we don't have to worry about the model being verbose in it's response, as we've communicated
that the text between the `#!xml <fun-fact></fun-fact>` tags is the relevent place to put it's answer.

### Strict Parsing

In the example above, we don't handle the case where the model fails to properly conform to our
desired output structure. If the last message content is invalid in some way, our call to `parse`
will result in an exception from rigging. Rigging is designed at it's core to manage this process, 
and we have a few options:

1. We can make the parsing optional by switching to [`.try_parse()`][rigging.message.Message.try_parse]. The type
   of the return value with automatically switch to `#!python FunFact | None` and you can handle cases
   where parsing failed.
2. We can extend our pending chat with [`.until_parsed_as()`][rigging.chat.PendingChat] which will cause the
   `run()` function to internally check if parsing is succeeding before returning the chat back to you.

=== "Option 1 - Trying"

    ```py hl_lines="5"
    chat = rg.get_generator('gpt-3.5-turbo').chat(
        f"Provide a fun fact between {FunFact.xml_example()} tags."
    ).run()

    fun_fact = chat.last.try_parse(FunFact) # fun_fact might now be None

    print(fun_fact or "Failed to get fact")
    ```

=== "Option 2 - Until"

    ```py hl_lines="4"
    chat = (
        rg.get_generator('gpt-3.5-turbo')
        .chat(f"Provide a fun fact between {FunFact.xml_example()} tags.")
        .until_parsed_as(FunFact)
        .run()
    )

    fun_fact = chat.last.parse(FunFact) # This call should never fail

    print(fun_fact or "Failed to get fact")
    ```

    A couple of comments regarding this structure:

    1. We still have to call `parse` on the message despite use using `until_parsed_as`. This is
    a limitation of type hinting as we'd have to turn every `PendingChat` and `Chat` into a generic
    which could carry types forward. It's a small price for big code complexity savings.
    2. Internally, the generation code inside `PendingChat` will attempt to re-generate until
    the LLM correctly produces a parsable input, up until a maximum number of "rounds" is reached.
    This process is configurable with the arguments to all [`until`][rigging.chat.PendingChat.until_parsed_as]
    or [`using`][rigging.chat.PendingChat.using] functions.

### Parsing Many Models

Assuming we wanted to extend our example to produce a set of interesting facts, we have a couple of options:

1. Simply use [`run_many()`][rigging.chat.PendingChat.run_many] and generate N examples individually
2. Rework our code slightly and let the model provide us multiple facts at once.

=== "Option 1 - Multiple Generations"

    ```py
    chats = rg.get_generator('gpt-3.5-turbo').chat(
        f"Provide a fun fact between {FunFact.xml_example()} tags."
    ).run_many(3)

    for chat in chats:
        print(chat.last.parse(FunFact).fact)
    ```

=== "Option 2 - Inline Set"

    ```py
    chat = rg.get_generator('gpt-3.5-turbo').chat(
        f"Provide a 3 fun facts each between {FunFact.xml_example()} tags."
    ).run()

    for fun_fact in chat.last.parse_set(FunFact):
        print(fun_fact.fact)
    ```

### Keep Going

Check out the **[topics section](topics/workflow.md)** for more in-depth explanations and examples.