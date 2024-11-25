import asyncio
import base64
import os
import typing as t

from pydantic import ConfigDict, StringConstraints

import rigging as rg
from rigging import logger
from rigging.generator import GenerateParams, Generator, register_generator

logger.enable("rigging")

MAX_TOKENS = 8000
TRUNCATION_WARNING = "\n\n**Note**: Due to the large size of this diff, some content has been truncated."
str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]


class PRDiffData(rg.Model):
    """XML model for PR diff data"""

    content: str_strip = rg.element()

    @classmethod
    def xml_example(cls) -> str:
        return """<diff><content>example diff content</content></diff>"""


class PRDecorator(Generator):
    """Generator for creating PR descriptions"""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    api_key: str = ""
    max_tokens: int = MAX_TOKENS

    def __init__(self, model: str, params: rg.GenerateParams) -> None:
        api_key = params.extra.get("api_key")
        if not api_key:
            raise ValueError("api_key is required in params.extra")

        super().__init__(model=model, params=params, api_key=api_key)
        self.api_key = api_key
        self.max_tokens = params.max_tokens or MAX_TOKENS

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[rg.Message]],
        params: t.Sequence[GenerateParams],
    ) -> t.Sequence[rg.GeneratedMessage]:
        responses = []
        for message_seq, p in zip(messages, params):
            base_generator = rg.get_generator(self.model, params=p)
            llm_response = await base_generator.generate_messages([message_seq], [p])
            responses.extend(llm_response)
        return responses


register_generator("pr_decorator", PRDecorator)


async def generate_pr_description(diff_text: str) -> str:
    """Generate a PR description from the diff text"""
    diff_tokens = len(diff_text) // 4
    if diff_tokens >= MAX_TOKENS:
        char_limit = (MAX_TOKENS * 4) - len(TRUNCATION_WARNING)
        diff_text = diff_text[:char_limit] + TRUNCATION_WARNING

    diff_data = PRDiffData(content=diff_text)
    params = rg.GenerateParams(
        extra={
            "api_key": os.environ["OPENAI_API_KEY"],
            "diff_text": diff_text,
        },
        temperature=0.7,
        max_tokens=500,
    )

    generator = rg.get_generator("pr_decorator!gpt-4-turbo-preview", params=params)
    prompt = f"""You are a helpful AI that generates clear and concise PR descriptions.
    Analyze the provided diff between {PRDiffData.xml_example()} tags and create a summary using exactly this format:

    ### PR Summary

    #### Overview of Changes
    <overview paragraph>

    #### Key Modifications
    1. **<modification title>**: <description>
    2. **<modification title>**: <description>
    3. **<modification title>**: <description>
    (continue as needed)

    #### Potential Impact
    - <impact point 1>
    - <impact point 2>
    - <impact point 3>
    (continue as needed)

    Here is the PR diff to analyze:
    {diff_data.to_xml()}"""

    chat = await generator.chat(prompt).run()
    return chat.last.content.strip()


async def main():
    """Main function for CI environment"""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set")

    try:
        diff_text = os.environ.get("GIT_DIFF", "")
        if not diff_text:
            raise ValueError("No diff found in GIT_DIFF environment variable")

        try:
            diff_text = base64.b64decode(diff_text).decode("utf-8")
        except Exception:
            padding = 4 - (len(diff_text) % 4)
            if padding != 4:
                diff_text += "=" * padding
            diff_text = base64.b64decode(diff_text).decode("utf-8")

        logger.debug(f"Processing diff of length: {len(diff_text)}")
        description = await generate_pr_description(diff_text)

        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("content<<EOF\n")
            f.write(description)
            f.write("\nEOF\n")
            f.write(f"debug_diff_length={len(diff_text)}\n")
            f.write(f"debug_description_length={len(description)}\n")
            debug_preview = description[:500]
            f.write("debug_preview<<EOF\n")
            f.write(debug_preview)
            f.write("\nEOF\n")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
