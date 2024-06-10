# Prompt Functions

Defining prompts as python functions is a great abstraction on top of chat pipelines, allowing you
to leverage parsing logic and type hints to define a code-less function that will use a generator
underneath.

## Defining Prompts

A [`Prompt`][rigging.prompt.Prompt] is typically created using one of the following decorators:

- [`@rg.prompt`][rigging.prompt.prompt] - Optionally takes a generator id, generator, or pipeline.
- [`@generator.prompt`][rigging.generator.Generator.prompt] - Use this generator when executing the prompt.
- [`@pipeline.prompt`][rigging.chat.ChatPipeline.prompt] - Use this pipeline when executing the prompt.

!!! note "Async Conversion"

    Prompt functions can be defined with or without the `async` keyword, but they will always be
    represented as async calls once wrapped based on their connection to chat pipelines.

    In other words, wrapping a synchronous function with `@rg.prompt` will result in an async
    callable.

=== "Generator ID"

    ```py
    import rigging as rg

    @rg.prompt(generator_id="mistral/mistral-medium-latest")
    def summarize(text: str) -> str:
        """Summarize this text."""

    ```

=== "Generator Instance"

    ```py
    import rigging as rg

    generator = rg.get_generator("mistral/mistral-medium-latest")

    @generator.prompt
    def summarize(text: str) -> str:
        """Summarize this text."""

    ```

=== "Chat Pipeline"

    ```py
    import rigging as rg

    pipeline = (
        rg.get_pipeline("mistral/mistral-medium-latest")
        .chat([
            {"role": "system", "content": "Respond like a pirate."}
        ])
    )

    @pipeline.prompt
    def summarize(text: str) -> str:
        """Summarize this text."""

    ```

=== "Unbound"

    ```py
    import rigging as rg

    @rg.prompt
    def summarize(text: str) -> str:
        """Summarize this text."""

    ```

Prompts are optionally bound to a pipeline/generator underneath, hence the generator and pipeline
decorator variants, but they don't have to be. We refer to bound prompts as "standalone", because
they can be executed directly as functions. Otherwise, you are required to use 
[`ChatPipeline.run_prompt`][rigging.chat.ChatPipeline.run_prompt] to execute the prompt.

### Templates and Docstrings

Underneath, the function signature will be analyzed for inputs and outputs, and the docstring
will be used to create a final jinja2 template which will be used as the prompt text. You can
always access this [`.template`][rigging.prompt.Prompt.template] attribute to inspect how the prompt
will be formatted.

Here are the general processing docstring rules:

- Any docstring content will always be at the top of the prompt.
- If no docstring is provided, a generic one will be substituted.
- Any inputs not explicitly defined in the docstring will be appended after the docstring.

=== "No Docstring"

    ```py
    import rigging as rg

    @rg.prompt
    def summarize(text: str) -> str:
        ...

    print(summarize.template)

    # Convert the following inputs to outputs (summarize).
    #
    # {{ text }}
    #
    # Produce the following output:
    #
    # <str></str>
    ```

=== "Basic Docstring"

    ```py
    import rigging as rg

    @rg.prompt
    def summarize(text: str) -> str:
        """Summarize this text."""

    print(summarize.template)

    # Summarize this text.
    #
    # {{ text }}
    #
    # Produce the following output:
    #
    # <str></str>
    ```

=== "Inputs in Docstring"

    ```py
    import rigging as rg

    @rg.prompt
    def summarize(text: str) -> str:
        """
        Summarize the following:
        
        {{ text }}
        
        Be concise.
        """

    print(summarize.template)

    # Summarize the following:
    #
    # {{ text }}
    #
    # Be concise.
    #
    # Produce the following output:
    #
    # <str></str>
    ```

In other words, you can define how inputs will be included in the prompt by passing them inside
the docstring in jinja2 formats, or let Rigging handle this for you by ommiting them.

### Outputs and Context

In the example above, you'll notice that defining our function to output a `str` results
in the following text be appended to the prompt template:

```py
# Produce the following output:
#
# <str></str>
```

This is pretty light on context, and we can improve this by updating our signature
with a [`Ctx`][rigging.prompt.Ctx] annotation:

```py
from typing import Annotated
import rigging as rg

summary = Annotated[str, rg.Ctx(tag="summary", example="[2-3 sentences]")]

@rg.prompt
def summarize(text: Annotated[str, rg.Ctx(tag="long-text")]) -> summary:
    """Summarize this text."""

print(summarize.template)

# Summarize this text.
#
# {{ long_text }}
#
# Produce the following output:
#
# <summary>[2-3 sentences]</summary>
```

We can apply [`Ctx`][rigging.prompt.Ctx] annotations to any of the inputs and outputs
of a prompt. We can override the xml tag, provide an example, and add prefix text.

Output processing is optional, and can be omitted by returning a [`Chat`][rigging.chat.Chat] object
from the wrapped function. This allows you to do with the generated output as you please.

```py
import rigging as rg

@rg.prompt
def summarize(text: str) -> rg.Chat:
    """Summarize this text."""

print(summarize.template)

# Summarize this text.
#
# {{ text }}
```

### Complex Outputs

You can also define more complex outputs by using a rigging model, list, tuple, or dataclass. Not every
construction will be supported, and we attempt to pre-validate the output structure to ensure it can be
processed correctly.

=== "Model"

    ```py
    import rigging as rg

    class User(rg.Model):
        name: str
        email: str
        age: int

    @rg.prompt
    def generate_user() -> User:
        """Generate a fake test user."""

    print(generate_user.template)

    # Generate a fake test user.
    #
    # Produce the following output:
    #
    # <user>
    #   <name/>
    #   <email/>
    #   <age/>
    # </user>
    ```

=== "List of Models"

    ```py
    import rigging as rg

    class User(rg.Model):
        name: str
        email: str
        age: int

    @rg.prompt
    def generate_users(count: int = 3) -> list[User]:
        """Generate fake test users."""

    print(generate_users.template)

    # Generate fake test users.
    #
    # {{ count }}
    #
    # Produce the following output for each item:
    #
    # <user>
    #   <name/>
    #   <email/>
    #   <age/>
    # </user>
    ```

=== "Dataclass"

    ```py
    from dataclasses import dataclass
    import rigging as rg

    @dataclass
    class User:
        name: str
        email: str
        age: int

    @rg.prompt
    def generate_user() -> User:
        """Generate a fake test user."""

    print(generate_user.template)

    # Generate a fake test user.
    #
    # Produce the following outputs:
    #
    # <name></name>
    #
    # <email></email>
    #
    # <age></age>
    ```

=== "Tuple"

    ```py
    import rigging as rg

    name = Annotated[str, rg.Ctx(tag="name")]
    email = Annotated[str, rg.Ctx(tag="email")]
    age = Annotated[int, rg.Ctx(tag="age")]

    @rg.prompt
    def generate_user() -> tuple[name, email, age]:
        """Generate a fake test user."""

    print(generate_user.template)

    # Generate a fake test user.
    #
    # Produce the following outputs:
    #
    # <name></name>
    #
    # <email></email>
    #
    # <age></age>
    ```

You can also embedd a [`Chat`][rigging.chat.Chat] object inside a some objects, which
will be excluded from any prompt guidance, but supplied the value when the prompt
is executed. This is great for gathering both structured data and the original chat. 

=== "Chat in Tuple"

    ```py
    import rigging as rg

    joke = Annotated[str, rg.Ctx(tag="joke")]

    @rg.prompt
    def tell_joke() -> tuple[joke, rg.Chat]:
        """Tell a joke."""

    print(tell_joke.template)

    # Tell a joke.
    #
    # Produce the following output:
    #
    # <joke></joke>
    ```

=== "Chat in Dataclass"

    ```py
    from dataclasses import dataclass
    import rigging as rg

    @dataclass
    class Joke:
        setup: str
        punchline: str
        chat: rg.Chat

    @rg.prompt
    def tell_joke() -> Joke:
        """Tell a joke."""

    print(tell_joke.template)

    # Tell a joke.
    #
    # Produce the following outputs:
    #
    # <setup></setup>
    #
    # <punchline></punchline>
    ```

## Rendering Prompts

In addition to templates, you can use [`.render`][rigging.prompt.Prompt.render] with valid
inputs to view the exact prompt as it will be sent to a generator. You can also use this
to pass your prompt into pipelines at your discretion.

```py
from typing import Annotated
import rigging as rg

email = Annotated[str, rg.Ctx(tag="email")]

@rg.prompt
def convert_to_email(name: str, top: int = 5) -> list[email]:
    """Convert this name into the best {{ top }} email addresses."""

print(convert_to_email.render("John Doe"))

# Convert this name into the best 5 email addresses.
#
# <name>John Doe</name>
#
# Produce the following output for each item:
#
# <email></email>
```

## Running Prompts

Prompt objects expose the following methods for execution:

- [`Prompt.run()`][rigging.prompt.Prompt.run] (Aliased with `__call__`)
- [`Prompt.run_many()`][rigging.prompt.Prompt.run_many]
- [`Prompt.run_over()`][rigging.prompt.Prompt.run_over]

*(Available if the prompt was supplied/bonded to a pipeline or generator)*

You can also run a prompt with a specific `ChatPipeline` by passing it to any of:

- [`ChatPipeline.run_prompt()`][rigging.chat.ChatPipeline.run_prompt]
- [`ChatPipeline.run_prompt_many()`][rigging.chat.ChatPipeline.run_prompt_many]
- [`ChatPipeline.run_prompt_over()`][rigging.chat.ChatPipeline.run_prompt_over]

!!! Note "Pipeline Context"

    Everything configured on a pipeline or generator will be used when running
    the prompt. Watch/Then/Map callbacks, tools, and generate params can all
    be used to alter the behavior of the prompt.

    In general, you should consider prompts as producers of user messages, which
    will be passed to [`.fork()`][rigging.chat.ChatPipeline.fork], then handle
    the parsing of outputs.

=== "Run Standalone"

    ```py
    import rigging as rg

    @rg.prompt(generator_id="claude-3-sonnet-20240229")
    def write_code(description: str, language: str = "python") -> str:
        """Write a single function."""

    code = await write_code("Calculate the factorial of a number.")
    ```

=== "Run with Pipeline"

    ```py
    from typing import Annotated
    import rigging as rg

    pipeline = (
        rg.get_generator("claude-3-sonnet-20240229")
        .chat([
            {"role": "system", "content": "You are a senior software developer"}
        ])
    )

    code_str = Annotated[str, rg.Ctx(tag="code")]

    @rg.prompt
    def write_code(description: str, language: str = "python") -> code_str:
        """Write a single function."""

    code = await pipeline.run_prompt(write_code, "Calculate the factorial of a number.")
    ```

=== "Run Manually"

    ```py
    import rigging as rg

    pipeline = (
        rg.get_generator("claude-3-sonnet-20240229")
        .chat([
            {"role": "system", "content": "You are a senior software developer"}
        ])
    )

    @rg.prompt
    def write_code(description: str, language: str = "python") -> rg.Chat:
        """Write a single function."""

    prompt = write_code.render("Calculate the factorial of a number.")

    chat = await pipeline.fork(prompt).run()
    ```