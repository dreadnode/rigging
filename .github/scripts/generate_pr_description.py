# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rigging",
#     "typer",
# ]
# ///

import asyncio
import subprocess
import typing as t

import typer

import rigging as rg

TRUNCATION_WARNING = "\n---\n**Note**: Due to the large size of this diff, some content has been truncated."


@rg.prompt
def generate_pr_description(diff: str) -> t.Annotated[str, rg.Ctx("markdown")]:  # type: ignore[empty-body]
    """
    Analyze the provided git diff and create a PR description in markdown format.

    <guidance>
    - Keep the summary concise and informative.
    - Use bullet points to structure important statements.
    - Focus on key modifications and potential impact - if any.
    - Do not add in general advice or best-practice information.
    - Write like a developer who authored the changes.
    - Prefer flat bullet lists over nested.
    - Do not include any title structure.
    </guidance>
    """


def get_diff(target_ref: str, source_ref: str) -> str:
    """
    Get the git diff between two branches.
    """

    merge_base = subprocess.run(
        ["git", "merge-base", source_ref, target_ref],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    diff_text = subprocess.run(
        ["git", "diff", merge_base],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return diff_text


def main(
    target_ref: str,
    source_ref: str = "HEAD",
    generator_id: str = "openai/gpt-4o-mini",
    max_diff_lines: int = 1000,
) -> None:
    """
    Use rigging to generate a PR description from a git diff.
    """

    diff = get_diff(target_ref, source_ref)
    diff_lines = diff.split("\n")
    if len(diff_lines) > max_diff_lines:
        diff = "\n".join(diff_lines[:max_diff_lines]) + TRUNCATION_WARNING

    description = asyncio.run(generate_pr_description.bind(generator_id)(diff))
    print(description)


if __name__ == "__main__":
    typer.run(main)
