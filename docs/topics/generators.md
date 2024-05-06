!!! note
    This content is currently being refactored

### Overload Generation Params

```python
import rigging as rg

pending = rg.get_generator("gpt-3.5-turbo,max_tokens=50").chat([
    {"role": "user", "content": "Say a haiku about boats"},
])

for temp in [0.1, 0.5, 1.0]:
    print(pending.overload(temperature=temp).run().last.content)

```

### Custom Generator

Any custom generator simply needs to implement a `complete` function, and 
then it can be used anywhere inside rigging.

```python
class Custom(Generator):
    # model: str
    # api_key: str
    # params: GeneratorParams
    
    custom_field: bool

    def complete(
        self,
        messages: t.Sequence[rg.Message],
        overloads: GenerateParams = GenerateParams(),
    ) -> rg.Message:
        # Access self vars where needed
        api_key = self.api_key
        model_id = self.model

        # Merge in args for API overloads
        marged: dict[str, t.Any] = self._merge_params(overloads)

        # response: str = ...

        return rg.Message("assistant", response)


generator = Custom(model='foo', custom_field=True)
generator.chat(...)
```

*Note: we currently don't have anyway to "register" custom generators for `get_generator`.*