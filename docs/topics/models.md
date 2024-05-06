!!! note
    This content is currently being refactored

### Model Parsing

```python
import rigging as rg

class Answer(rg.Model):
    content: str

chat = (
    rg.get_generator("claude-3-haiku-20240307")
    .chat([
        {"role": "user", "content": f"Say your name between {Answer.xml_tags()}."},
    ])
    .until_parsed_as(Answer)
    .run()
)

answer = chat.last.parse(Answer)
print(answer.content)

# "Claude"

print(f"{chat.last!r}")

# Message(role='assistant', parts=[
#   ParsedMessagePart(model=Answer(content='Claude'), ref='<answer>Claude</answer>')
# ], content='<Answer>Claude</Answer>')

chat.last.content = "new content" # Updating content strips parsed parts
print(f"{chat.last!r}")

# Message(role='assistant', parts=[], content='new content')
```

### Mutliple Models

```python
import rigging as rg

class Joke(rg.Model):
    content: str

chat = (
    rg.get_generator("claude-2.1")
    .chat([{
        "role": "user", 
        "content": f"Provide 3 short jokes each wrapped with {Joke.xml_tags()} tags."},
    ])
    .run()
)

jokes = chat.last.parse_set(Joke)

# [
#     Joke(content="Why don't eggs tell jokes? They'd crack each other up!"),
#     Joke(content='What do you call a bear with no teeth? A gummy bear!'),
#     Joke(content='What do you call a fake noodle? An Impasta!')
# ]
```

### Complex Models

```python
import rigging as rg

class Inner(rg.Model):
    type: str = rg.attr()
    content: str

class Outer(rg.Model):
    name: str = rg.attr()
    inners: list[Inner] = rg.element()

outer = Outer(name="foo", inners=[
    Inner(type="cat", content="meow"),
    Inner(type="dog", content="bark")
])

print(outer.to_pretty_xml())

# <outer name="foo">
#    <inner type="cat">meow</inner>
#    <inner type="dog">bark</inner>
# </outer>
```