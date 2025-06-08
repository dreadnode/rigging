<p align="center">
    <img
    src="https://d1lppblt9t2x15.cloudfront.net/logos/5714928f3cdc09503751580cffbe8d02.png"
    alt="Logo"
    align="center"
    width="144px"
    height="144px"
    />
</p>

<h3 align="center">
Flexible LLM library for code and agents
</h3>

<h4 align="center">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/rigging">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/rigging">
    <img alt="GitHub License" src="https://img.shields.io/github/license/dreadnode/rigging">
    <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/dreadnode/rigging/ci.yml">
</h4>

</br>

Rigging is a lightweight LLM framework to make using language models in production code as simple and effective as possible. Here are the highlights:

- **Structured Pydantic models** can be used interchangeably with unstructured text output.
- LiteLLM as the default generator giving you **instant access to a huge array of models**.
- Define prompts as python functions with **type hints and docstrings**.
- Simple **tool use**, even for models which don't support them at the API.
- Store different models and configs as **simple connection strings** just like databases.
- Integrated tracing support with [Logfire](https://logfire.pydantic.dev/docs/).
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

If you want to build from source:
```bash
cd rigging/
poetry install
```

## Supported LLMs

Rigging will run just about any language model:

- Any model from [**LiteLLM**](https://litellm.vercel.app/docs/providers)
- Any model from [**vLLM**](https://docs.vllm.ai/en/latest/models/supported_models.html)
- Any model from [**transformers**](https://huggingface.co/docs/transformers/)

### API Keys

Pass the `api_key` in an generator id or use standard environment variables.

```py
rg.get_generator("gpt-4-turbo,api_key=...")
```

```bash
export OPENAI_API_KEY=...
export MISTRAL_API_KEY=...
export ANTHROPIC_API_KEY=...
...
```

Check out [the docs](https://docs.dreadnode.io/open-source/rigging/topics/generators#api-keys) for more.

## Getting Started

**Check out the guide [in the docs](https://docs.dreadnode.io/open-source/rigging/intro#getting-started)**

1. **Get a generator** using a connection string.
2. Build a **chat** or **completion** pipeline
3. **Run** the pipeline and get the output.

```py
import rigging as rg
import asyncio

async def main():
    # 1 - Get a generator
    generator = rg.get_generator("claude-3-sonnet-20240229")

    # 2 - Build a chat pipeline
    pipeline = generator.chat(
        [
            {"role": "system", "content": "Talk like a pirate."},
            {"role": "user", "content": "Say hello!"},
        ]
    )

    # 3 - Run the pipeline
    chat = await pipeline.run()
    print(chat.conversation)

# Run the main function
asyncio.run(main())

# [system]: Talk like a pirate.
# [user]: Say hello!
# [assistant]: Ahoy, matey! Here be the salty sea dog ready to trade greetings wit' ye. Arrr!
```

Want more?

- Use [structured pydantic parsing](https://docs.dreadnode.io/open-source/rigging/topics/chats-and-messages#parsed-parts)
- Check out [raw completions](https://docs.dreadnode.io/open-source/rigging/topics/completions/)
- Give the LLM [access to tools](https://docs.dreadnode.io/open-source/rigging/topics/tools/)
- Track behavior with [tracing](https://docs.dreadnode.io/open-source/rigging/topics/tracing/)
- Play with [generation params](https://docs.dreadnode.io/open-source/rigging/topics/generators/#overload-generation-params)
- Use [callbacks in the pipeline](https://docs.dreadnode.io/open-source/rigging/topics/callbacks-and-mapping/)
- Scale up with [iterating and batching](https://docs.dreadnode.io/open-source/rigging/topics/iterating-and-batching/)
- Save your work with [serialization](https://docs.dreadnode.io/open-source/rigging/topics/serialization/)

## Examples

- Basic interactive chat: [**chat.py**](examples/chat.py)
- Jupyter code interpreter: [**jupyter.py**](examples/jupyter.py)
- OverTheWire Bandit Agent: [**bandit.py**](examples/bandit.py)
- Damn Vulnerable Restaurant Agent: [**dvra.py**](examples/dvra.py)
- RAG Pipeline: [**rag.py**](examples/rag.py) (from [kyleavery](https://github.com/kyleavery/))
- Integrating dreadnode-owned [robopages](https://github.com/dreadnode/robopages-cli) as a tool server (basic nmap scan example): [**rigging_example.py**](https://github.com/dreadnode/robopages-cli/blob/main/examples/rigging_example.py)

## Documentation

**[docs.dreadnode.io](https://docs.dreadnode.io/open-source/rigging/intro)** has everything you need.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=dreadnode/rigging&type=Date)](https://star-history.com/#dreadnode/rigging&Date)
