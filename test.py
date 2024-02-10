import rigging as rg


class Reasoning(rg.Model):
    content: str


generator = rg.get_generator("claude-2.1")

meaning = generator.chat(
    [
        {
            "role": "user",
            "content": "What is the meaning of life in one sentence? "
            f"Document your reasoning between {Reasoning.xml_tags()} tags.",
        },
    ]
).run()

# Gracefully attempt to parse and deal
# with missing models as None

reasoning = meaning.last.try_parse(Reasoning)
if reasoning:
    print("reasoning:", reasoning.content.strip())

# Strip parsed content to avoid sharing
# previous thoughts with the model.

without_reasons = meaning.strip(Reasoning)
print("meaning of life:", without_reasons.last.content.strip())

# follow_up = without_thoughts.continue_(...)
