Rigging is a lightweight LLM framework built on Pydantic XML. The goal is to make leveraging language models in production code as simple and effective as possible. Here are the highlights:

- **Structured Pydantic models** can be used interchangably with unstructured text output.
- LiteLLM as the default generator giving you **instant access to a huge array of models**.
- Define prompts as python functions with **type hints and docstrings**.
- Simple **tool calling** abilities for models which don't natively support it.
- Store different models and configs as **simple connection strings** just like databases.
- Chat templating, forking, continuations, generation parameter overloads, stripping segments, etc.
- Async batching and fast iterations for **large scale generation**.
- Metadata, callbacks, and data format conversions.
- Modern python with type hints, async support, pydantic validation, serialization, etc.

```py
import rigging as rg

@rg.prompt(generator_id="gpt-4")
async def get_authors(count: int = 3) -> list[str]:
    """Provide famous authors."""

print(await get_authors())

# ['William Shakespeare', 'J.K. Rowling', 'Jane Austen']
```

Rigging is built by [**dreadnode**](https://dreadnode.io) where we use it daily.

## Installation

We publish every version to Pypi:
```bash
pip install rigging
```

If you want all the extras (vLLM, transformers, examples), just specify the `all` extra:
```bash
pip install rigging[all]
```

If you want to build from source:
```bash
cd rigging/
poetry install
```

## Migration Guides

- **[Migrating from v1.x to v2.x](topics/migrations.md#migrating-from-v1x-to-v2x)**

## Getting Started

Rigging is a flexible library built on other flexible libraries. As such it might take a bit to warm
up to it's interfaces provided the many ways you can accomplish your goals. However, the code is well documented
and topic pages and source are a great places to step in/out of as you explore.

??? tip "IDE Setup"

    Rigging has been built with full type support which provides clear guidance on what
    methods return what types, and when they return those types. It's recommended that you
    operate in a development environment which can take advantage of this information.
    Rigging will almost "fall" into place and you won't be guessing about
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
pipeline = generator.chat(
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
)
chat = await pipeline.run() # (3)!
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
3. From version 2 onwards, Rigging is fully async. You can use `await` to trigger generation
   and get your results, or use [`await_`][rigging.util.await_].


Generators have an easy [`chat()`][rigging.generator.Generator.chat] method which you'll
use to initiate the conversations. You can supply messages in many different forms from
dictionary objects, full [`Message`][rigging.message.Message] classes, or a simple `str`
which will be converted to a user message.

```py hl_lines="4-9"
import rigging as rg

generator = rg.get_generator("claude-3-sonnet-20240229")
pipeline = generator.chat( # (1)!
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
)
chat = await pipeline.run()
print(chat.all)
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
#   Message(role='assistant', parts=[], content='Hello! How can I help you today?'),
# ]
```

1. [`generator.chat`][rigging.generator.Generator.chat] is actually just a helper for
   [`chat(generator, ...)`][rigging.generator.chat], they do the same thing.

??? note "ChatPipeline vs Chat"

    You'll notice we name the result of `chat()` as `pipeline`. The naming might be confusing,
    but chats go through 2 phases. We first stage them into a pipeline, where we operate
    and prepare them before we actually trigger generation with `run()`.

    Calling `.chat()` doesn't trigger any generation, but calling any of these run methods will:

    - [rigging.chat.ChatPipeline.run][]
    - [rigging.chat.ChatPipeline.run_many][]
    - [rigging.chat.ChatPipeline.run_batch][]
    - [rigging.chat.ChatPipeline.run_over][]

In this case, we have nothing additional we want to add to our chat pipeline, and we are only interested
in generating exactly one response message. We simply call [`.run()`][rigging.chat.ChatPipeline.run] to
execute the generation process and collect our final [`Chat`][rigging.chat.Chat] object.

```py hl_lines="10-11"
import rigging as rg

generator = rg.get_generator("claude-3-sonnet-20240229")
pipeline = generator.chat(
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
)
chat = await pipeline.run()
print(chat.all)
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
#   Message(role='assistant', parts=[], content='Hello! How can I help you today?'),
# ]
```

View more about Chat objects and their properties [over here][rigging.chat.Chat]. In general, chats
give you access to exactly what messages were passed into a model, and what came out the other side.

### Prompts

Operating chat pipelines manually is very flexible, but can feel a bit verbose. Rigging supports
the concept of "prompt functions" where you to define the interaction with an LLM as a python function
signature, and convert that to a callable object which abstracts the pipeline away from you.

=== "From ID"

    ```py
    import rigging as rg

    @rg.prompt(generator_id="claude-3-sonnet-20240229")
    async def say_hello(name: str) -> rg.Chat:
        """Say hello to {{ name }}"""

    chat = await say_hello("Harry")
    ```

=== "From Generator"

    ```py
    import rigging as rg

    generator = rg.get_generator("claude-3-sonnet-20240229")

    @generator.prompt
    async def say_hello(name: str) -> rg.Chat:
        """Say hello to {{ name }}"""

    chat = await say_hello("Harry")
    ```

=== "From Pipeline"

    ```py
    import rigging as rg

    generator = rg.get_generator("claude-3-sonnet-20240229")
    pipeline = generator.chat([
        {"role": "system", "content": "Talk like a pirate."}
    ])

    @pipeline.prompt
    async def say_hello(name: str) -> rg.Chat:
        """Say hello to {{ name }}"""

    chat = await say_hello("Harry")
    ```

Prompts are very powerful. You can take control over any of the inputs in your docstring, gather
outputs as structured objects, lists, dataclasses, and collect the underlying Chat object, etc.

Check out [Prompt Functions](topics/prompt-functions.md) for more information.

### Conversations

Both [`ChatPipeline`][rigging.chat.ChatPipeline] and [`Chat`][rigging.chat.Chat] objects provide freedom
for forking off the current state of messages, or continuing a stream of messages after generation has occured.

In general:

- [`ChatPipeline.fork`][rigging.chat.ChatPipeline.fork] will clone the current chat pipeline and let you maintain
  both the new and original object for continued processing.
- [`Chat.fork`][rigging.chat.Chat.fork] will produce a fresh `ChatPipeline` from all the messages prior to the
  previous generation (useful for "going back" in time).
- [`Chat.continue_`][rigging.chat.Chat.continue_] is similar to `fork` (actually a wrapper) which tells `fork` to
  include the generated messages as you move on (useful for "going forward" in time).

In other words, the abstraction of going back and forth in a "conversation" would be continuously calling
[`Chat.continue_`][rigging.chat.Chat.continue_] after each round of generation.

```py
import rigging as rg

generator = rg.get_generator("gpt-3.5-turbo")
chat = generator.chat("Hello, how are you?")

# We can fork before generation has occured
specific = await chat.fork("Be specific please.").run()
poetic = await chat.fork("Be as poetic as possible").with_(temperature=1.5).run() # (1)!

# We can also continue after generation
next_chat = poetic.continue_("That's good, tell me a joke") # (2)!

update = await next_chat.run()
```

1. In this case the temperature change will only be applied to the poetic path because `fork` has
   created a clone of our chat pipeline. 
2. For convience, we can usually just pass `str` objects in place of full messages, which underneath
   will be converted to a [`Message`][rigging.message.Message] object with the `user` role.

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

chat = await rg.get_generator('gpt-3.5-turbo').chat(
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

chat = await rg.get_generator('gpt-3.5-turbo').chat(
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

1. We can extend our chat pipeline with [`.until_parsed_as()`][rigging.chat.ChatPipeline] which will cause the
   [`run()`][rigging.chat.ChatPipeline.run] function to internally check if parsing is succeeding
   before returning the chat back to you.
2. We can make the parsing optional by switching to [`.try_parse()`][rigging.message.Message.try_parse]. The type
   of the return value with automatically switch to `#!python FunFact | None` and you can handle cases
   where parsing failed.

=== "Option 1 - Until Parsed As"

    ```py hl_lines="5"
    chat = (
        await
        rg.get_generator('gpt-3.5-turbo')
        .chat(f"Provide a fun fact between {FunFact.xml_example()} tags.")
        .until_parsed_as(FunFact)
        .run()
    )

    fun_fact = chat.last.parse(FunFact) # This call should never fail

    print(fun_fact or "Failed to get fact")
    ```

    !!! note "Double Parsing"
    
        We still have to call [`.parse()`][rigging.message.Message.parse] on the message despite
        using [`.until_parsed_as()`][rigging.chat.ChatPipeline.until_parsed_as]. This is
        a limitation of type hinting as we'd have to turn every `ChatPipeline` and `Chat` into a generic
        which could carry types forward. It's a small price for big code complexity savings. However,
        the use of [`.until_parsed_as()`][rigging.chat.ChatPipeline.until_parsed_as] **will** cause
        the generated messages to have parsed models in their [`.parts`][rigging.message.Message.parts].
        So if you don't need to access the typed object immediately, you can be confident serializing
        the chat object and the model will be there when you need it.

    !!! note "Max Rounds Concept"

        When control is passed into a chat pipeline with [`.until_parsed_as()`][rigging.chat.ChatPipeline.until_parsed_as],
        a callback is registered internally to operate during generation. When model output is received, the
        callback will attempt to parse, and if it fails, it will re-trigger generation with or without context depending
        on the [`attempt_recovery`][rigging.chat.ChatPipeline.until_parsed_as] parameter. This process will repeat
        until the model produces a valid output or the maximum number of "rounds" is reached.

        Often you might find yourself constantly getting [`ExhaustedMaxRoundsError`][rigging.error.ExhaustedMaxRoundsError]
        exceptions. This is usually a sign that the LLM doesn't have enough information about the desired output, or
        complexity in your model is too high. You have a few options for gracefull handling these situations:
        
        1. You can adjust the `max_rounds` as needed and try using `attempt_recovery`.
        2. Pass `allow_failed` to your [`run()`][rigging.chat.ChatPipeline.run] 
            method and check the [`.failed`][rigging.chat.Chat.failed] property after generation
        3. Use an external callback like [`.then()`][rigging.chat.ChatPipeline.then] to 
            get more external control over the process.

=== "Option 2 - Try Parse"

    ```py hl_lines="5"
    chat = await rg.get_generator('gpt-3.5-turbo').chat(
        f"Provide a fun fact between {FunFact.xml_example()} tags."
    ).run()

    fun_fact = chat.last.try_parse(FunFact) # fun_fact might now be None

    print(fun_fact or "Failed to get fact")
    ```

### Parsing Multiple Models

Assuming we wanted to extend our example to produce a set of interesting facts, we have a couple of options:

1. Simply use [`run_many()`][rigging.chat.ChatPipeline.run_many] and generate N examples individually
2. Rework our code slightly and let the model provide us multiple facts at once.

=== "Option 1 - Multiple Generations"

    ```py
    chats = await rg.get_generator('gpt-3.5-turbo').chat(
        f"Provide a fun fact between {FunFact.xml_example()} tags."
    ).run_many(3)

    for chat in chats:
        print(chat.last.parse(FunFact).fact)
    ```

=== "Option 2 - Inline Set"

    ```py
    chat = await rg.get_generator('gpt-3.5-turbo').chat(
        f"Provide a 3 fun facts each between {FunFact.xml_example()} tags."
    ).run()

    for fun_fact in chat.last.parse_set(FunFact):
        print(fun_fact.fact)
    ```

### Parsing with Prompts

The use of [`Prompt`][rigging.prompt.Prompt] functions can make parsing even easier. We can refactor
our previous example and have rigging parse out FunFacts directly for us:

=== "Multiple Generations"

    ```py
    import rigging as rg

    class FunFact(rg.Model):
        fact: str

    @rg.prompt(generator_id="gpt-3.5-turbo")
    def get_fun_fact() -> FunFact:
        """Provide a fun fact."""

    fun_facts = await get_fun_facts.run_many(3)
    ```

=== "Inline Set"

    ```py
    import rigging as rg

    class FunFact(rg.Model):
        fact: str

    @rg.prompt(generator_id="gpt-3.5-turbo")
    def get_fun_facts(count: int = 3) -> list[FunFact]:
        """Provide fun facts."""

    fun_facts = await get_fun_facts()
    ```

### Keep Going

Check out the **[topics section](topics/workflow.md)** for more in-depth explanations and examples.
