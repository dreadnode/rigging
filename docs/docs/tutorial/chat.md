### Basic Chats

```python
import rigging as rg

generator = rg.get_generator("claude-2.1")
chat = generator.chat(
    [
        {"role": "system", "content": "You are a wizard harry."},
        {"role": "user", "content": "Say hello!"},
    ]
).run()

print(chat.last)
# [assistant]: Hello!

print(f"{chat.last!r}")
# Message(role='assistant', parts=[], content='Hello!')

print(chat.prev)
# [
#   Message(role='system', parts=[], content='You are a wizard harry.'),
#   Message(role='user', parts=[], content='Say hello!'),
# ]

print(chat.json)
# [{ ... }]

```