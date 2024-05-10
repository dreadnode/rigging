# Generators

Underlying LLMs (or any function which completes text) is represented as a generator in Rigging.
They are typically instantiated using identifier strings and the [`get_generator`][rigging.generator.get_generator] function.
The base interface is flexible, and designed to support optimizations should the underlying mechanisms support it (batching
async, K/V cache, etc.)

## Identifiers

Much like database connection strings, Rigging generators can be represented as strings which define what provider, model, 
API key, generation params, etc. should be used. They are formatted as follows:

```
<provider>!<model>,<**kwargs>
```

- `provider` maps to a particular subclass of [`Generator`][rigging.generator.Generator].
- `model` is a any `str` value, typically used by the provider to indicate a specific LLM to target.
- `kwargs` are used to carry serialized [`GenerateParams`][rigging.generator.GenerateParams] to items like temp, stop tokens, etc.

The provider is optional and Rigging will fallback to `litellm`/[`LiteLLMGenerator`][rigging.generator.LiteLLMGenerator] by default.
You can view the [LiteLLM docs](https://docs.litellm.ai/docs/) for more information about supported model providers and parameters.

Here are some examples of valid identifiers:

```
gpt-3.5-turbo,temperature=0.5
openai/gpt-4,api_key=sk-1234
litellm!claude-3-sonnet-2024022
anthropic/claude-2.1,stop=output:;---,seed=1337
together_ai/meta-llama/Llama-3-70b-chat-hf
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

## Generator interface

::: rigging.generator.Generator
    options:
        show_source: false
        show_signature: false
        members:
        - generate_messages
        - generate_texts


## Overload Generation Params

When working with both [`PendingCompletion`][rigging.completion.PendingCompletion] and [`PendingChat`][rigging.chat.PendingChat], you
can overload and update any generation params by using the associated [`.with_()`][rigging.chat.PendingChat.with_] function. 

=== "with_() as keyword arguments"

    ```py
    import rigging as rg

    pending = rg.get_generator("gpt-3.5-turbo,max_tokens=50").chat([
        {"role": "user", "content": "Say a haiku about boats"},
    ])

    for temp in [0.1, 0.5, 1.0]:
        print(pending.with_(temperature=temp).run().last.content)
    ```

=== "with_() as `GenerateParams`"

    ```py
    import rigging as rg

    pending = rg.get_generator("gpt-3.5-turbo,max_tokens=50").chat([
        {"role": "user", "content": "Say a haiku about boats"},
    ])

    for temp in [0.1, 0.5, 1.0]:
        print(pending.with_(rg.GenerateParams(temperature=temp)).run().last.content)
    ```


## Writing a Generator

All generators should inherit from the [`Generator`][rigging.generator.Generator] base class, and
can elect to implement a series of messages, text, and async methods:

- [`def generate_messages(...)`][rigging.generator.Generator.generate_messages] - Used for [`PendingChat.run`][rigging.chat.PendingChat.run] variants.
- [`async def agenerate_messages(...)`][rigging.generator.Generator.agenerate_messages] - Used for [`PendingChat.arun`][rigging.chat.PendingChat.arun] variants.
- [`def generate_texts(...)`][rigging.generator.Generator.generate_texts] - Used for [`PendingCompletion.run`][rigging.completion.PendingCompletion.run] variants.
- [`async def agenerate_texts(...)`][rigging.generator.Generator.agenerate_texts] - Used for [`PendingCompletion.arun`][rigging.completion.PendingCompletion.arun] variants.

*If your generator doesn't implement a particular method like async or text completions, Rigging
will simply raise a `NotImplementedError` for you*

Generators operate in a batch context by default, taking in groups of message lists or texts. Whether
your implementation takes advantage of this batching is up to you, but where possible you
should be optimizing as much as possible.

```py
class Custom(Generator):
    # model: str
    # api_key: str
    # params: GeneratorParams
    
    custom_field: bool

    def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[Message]],
        params: t.Sequence[GenerateParams],
        *,
        prefix: t.Sequence[Message] | None = None, # (1)!
    ) -> t.Sequence[Message]:
        # If you aren't using prefix for any caching,
        # you'll frequently just concatenate it
        if prefix is not None:
            messages = [list(prefix) + list(messages) for messages in messages]
        
        # merge_with is an easy way to combine overloads
        params = [
            self.params.merge_with(p).to_dict() for p in params 
        ]

        # Access self vars where needed
        api_key = self.api_key
        model_id = self.model

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