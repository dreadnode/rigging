import asyncio

import click

import rigging as rg


async def main(generator_id: str, system_prompt: str) -> None:
    base_pipeline = rg.get_generator(generator_id).chat({"role": "system", "content": system_prompt})
    await rg.interact(base_pipeline)


@click.command()
@click.option(
    "-g",
    "--generator-id",
    type=str,
    required=True,
    help="Rigging generator identifier (gpt-4, mistral/mistral-medium, etc.)",
)
@click.option(
    "-s",
    "--system-prompt",
    type=str,
    default="You are a helpful assistant.",
    help="System prompt to use for the generator",
)
def cli(
    generator_id: str,
    system_prompt: str,
) -> None:
    """
    Rigging example of a basic terminal chat interaction.
    """

    asyncio.run(main(generator_id, system_prompt))


if __name__ == "__main__":
    cli()
