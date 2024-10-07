# Generators

Underlying LLMs (or any function which completes text) is represented as a generator in Rigging.
They are typically instantiated using identifier strings and the 
[`get_generator`][rigging.generator.get_generator] function.

The base interface is flexible, and designed to support optimizations should the
underlying mechanisms support it (batching async, K/V cache, etc.)

## Identifiers

Much like database connection strings, Rigging generators can be represented as
strings which define what provider, model,  API key, generation params, etc.
should be used. They are formatted as follows:

```
<provider>!<model>,<**kwargs>
```

- `provider` maps to a particular subclass of [`Generator`][rigging.generator.Generator].
- `model` is a any `str` value, typically used by the provider to indicate a specific LLM to target.
- `kwargs` are used to carry:
    1. Serialized [`GenerateParams`][rigging.generator.GenerateParams] fields like like temp, stop tokens, etc.
    2. Additional provider-specific attributes to set on the constructed generator class. For instance, you
       can set the [`LiteLLMGenerator.max_connections`][rigging.generator.litellm_.LiteLLMGenerator] property
       by passing `,max_connections=` in the identifier string.

The provider is optional and Rigging will fallback to 
[`litellm`](https://github.com/BerriAI/litellm)/[`LiteLLMGenerator`][rigging.generator.LiteLLMGenerator]
by default. You can view the [LiteLLM docs](https://docs.litellm.ai/docs/) for more
information about supported model providers and parameters.

Here are some examples of valid identifiers:

```
gpt-3.5-turbo,temperature=0.5
openai/gpt-4,api_key=sk-1234
litellm!claude-3-sonnet-2024022
anthropic/claude-2.1,stop=output:;---,seed=1337
together_ai/meta-llama/Llama-3-70b-chat-hf
openai/google/gemma-7b,api_base=https://integrate.api.nvidia.com/v1
```

Building generators from string identifiers is optional, but a convenient way to represent complex LLM configurations.

!!! tip "Back to Strings"

    Any generator can be converted back into an identifier using either [`to_identifier`][rigging.generator.Generator.to_identifier]
    or [`get_identifier`][rigging.generator.get_identifier].

    ```py
    generator = rg.get_generator("gpt-3.5-turbo,temperature=0.5")
    print(generator.to_identifier())
    # litellm!gpt-3.5-turbo,temperature=0.5
    ```

## API Keys

All generators carry a [`.api_key`][rigging.generator.Generator.api_key] attribute which can be set directly, or by
passing `,api_key=` as part of an identifier string. Not all generators will require one, but they are common enough
that we include the attribute as part of the base class.

Typically you will be using a library like LiteLLM underneath, and can simply use environment variables:

```bash
export OPENAI_API_KEY=...
export TOGETHER_API_KEY=...
export TOGETHERAI_API_KEY=...
export MISTRAL_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Rate Limits

Generators that leverage remote services (LiteLLM) expose properties for managing connection/request limits:

- [`LiteLLMGenerator.max_connections`][rigging.generator.litellm_.LiteLLMGenerator]
- [`LiteLLMGenerator.min_delay_between_requests`][rigging.generator.litellm_.LiteLLMGenerator]

However, a more flexible solution is [`ChatPipeline.wrap()`][rigging.chat.ChatPipeline.wrap] 
with a library like [**backoff**](https://github.com/litl/backoff) to catch
many, or specific errors, like rate limits or general connection issues.

```py
import rigging as rg

import backoff
import backoff.types


def on_backoff(details: backoff.types.Details) -> None:
    print(f"Backing off {details['wait']:.2f}s")

pipeline = (
    rg.get_generator("claude-3-haiku-20240307")
    .chat("Give me a 4 word phrase about machines.")
    .wrap(
        backoff.on_exception(
            backoff.expo,
            Exception,  # This should be scoped down
            on_backoff=on_backoff,
        )
    )
)

chats = await pipeline.run_many(50)
```

!!! note "Exception mess"

    You'll find that the exception consistency inside LiteLLM is quite poor.
    Different providers throw different types of exceptions for all kinds of
    status codes, response data, etc. With that said, you can typically find
    a target list that works well for your use-case.

## Local Models

Rigging supports local models via the [`ollama`](https://ollama.ai/) provider:

```py
import rigging as rg

# default to http://localhost:11434
ollama = rg.get_generator("ollama/llama3.1")
```

In the default configuration, the API base address is set to `http://localhost:11434`. It is possible to specify an alternative server by setting the OLLAMA_API_BASE environment variable:

```py
import os
import rigging as rg

os.environ['OLLAMA_API_BASE'] = 'http://192.168.0.10:11434'

ollama = rg.get_generator("ollama/llama3.1")
```

We also have experimental support for both [`vLLM`](https://docs.vllm.ai/en/latest/) 
and [`transformers`](https://huggingface.co/docs/transformers/index) generators for
loading and running local models. In general vLLM is more consistent with Rigging's
preferred API, but the dependency requirements are heavier.

Where needed, you can wrap an existing model into a rigging generator by using the
[`VLLMGenerator.from_obj()`][rigging.generator.vllm_.VLLMGenerator.from_obj] or
[`TransformersGenerator.from_obj()`][rigging.generator.transformers_.TransformersGenerator.from_obj] methods.
These are helpful for any picky model construction that might not play well with our rigging constructors. 

!!! note "Required Packages"

    The use of these generators requires the `vllm` and `transformers` packages to be installed.
    You can use `rigging[all]` to install them all at once, or pick your preferred package individually.

```py
import rigging as rg

tiny_llama = rg.get_generator(
    "vllm!TinyLlama/TinyLlama-1.1B-Chat-v1.0," \
    "gpu_memory_utilization=0.3," \
    "trust_remote_code=True"
)

llama_3 = rg.get_generator(
    "transformers!meta-llama/Meta-Llama-3-8B-Instruct"
)
```

See more about them below:

- [`vLLMGenerator`][rigging.generator.vllm_.VLLMGenerator]
- [`TransformersGenerator`][rigging.generator.transformers_.TransformersGenerator]

!!! tip "Loading and Unloading"

    You can use the [`Generator.load`][rigging.generator.Generator.load] and
    [`Generator.unload`][rigging.generator.Generator.unload] methods to better
    control memory usage. Local providers typically are lazy and load the model
    into memory only when first needed.

## Overload Generation Params

When working with both [`CompletionPipeline`][rigging.completion.CompletionPipeline] and 
[`ChatPipeline`][rigging.chat.ChatPipeline], you can overload and update any generation
params by using the associated [`.with_()`][rigging.chat.ChatPipeline.with_] function. 

=== "with_() as keyword arguments"

    ```py
    import rigging as rg

    pipeline = rg.get_generator("gpt-3.5-turbo,max_tokens=50").chat([
        {"role": "user", "content": "Say a haiku about boats"},
    ])

    for temp in [0.1, 0.5, 1.0]:
        chat = await pipeline.with_(temperature=temp).run()
        print(chat.last.content)
    ```

=== "with_() as `GenerateParams`"

    ```py
    import rigging as rg

    pipeline = rg.get_generator("gpt-3.5-turbo,max_tokens=50").chat([
        {"role": "user", "content": "Say a haiku about boats"},
    ])

    for temp in [0.1, 0.5, 1.0]:
        chat = await pipeline.with_(rg.GenerateParams(temperature=temp)).run()
        print(chat.last.content)
    ```

## Writing a Generator

All generators should inherit from the [`Generator`][rigging.generator.Generator] base class, and
can elect to implement handlers for messages and/or texts:

- [`async def generate_messages(...)`][rigging.generator.Generator.generate_messages] - Used for [`ChatPipeline.run`][rigging.chat.ChatPipeline.run] variants.
- [`async def generate_texts(...)`][rigging.generator.Generator.generate_texts] - Used for [`CompletionPipeline.run`][rigging.completion.CompletionPipeline.run] variants.

!!! note "Optional Implementation"

    If your generator doesn't implement a particular method like text completions, Rigging
    will simply raise a `NotImplementedError` for you. It's currently undecided whether generators
    should prefer to provide weak overloads for compatibility, or whether they should ignore methods
    which can't be used optimally to help provide clarity to the user about capability. You'll find
    we've opted for the former strategy in our generators.

Generators operate in a batch context by default, taking in groups of message lists or texts. Whether
your implementation takes advantage of this batching is up to you, but where possible you
should be optimizing as much as possible.

!!! tip "Generators are Flexible"

    Generators don't make any assumptions about the underlying mechanism that completes text.
    You might use a local model, API endpoint, or static code, etc. The base class is designed
    to be flexible and support a wide variety of use cases. You'll obviously find that the inclusion
    of `api_key`, `model`, and generation params are common enough that they are included in the base class.

```py
from rigging import Generator, GenerateParams, Message, GeneratedMessage

class Custom(Generator):
    # model: str
    # api_key: str
    # params: GeneratorParams
    
    custom_field: bool

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        # merge_with is an easy way to combine overloads
        params = [
            self.params.merge_with(p).to_dict() for p in params 
        ]

        # Access self vars where needed
        api_key = self.api_key
        model_id = self.model
        custom = self.custom_field
        
        # Build output messages with stop reason, usage, etc.
        # output_messages = ...

        return output_messages


generator = Custom(model='foo', custom_field=True)
generator.chat(...)
```

!!! tip "Registering Generators"

    Use the [`register_generator`][rigging.generator.register_generator] method to add your generator
    class under a custom provider id so it can be used with [`get_generator`][rigging.generator.get_generator].

    ```py
    import rigging as rg

    rg.register_generator('custom', Custom)
    custom = rg.get_generator('custom!foo')
    ```