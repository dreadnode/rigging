# Writing Models

Model definitions are at the core of Rigging, and provide an extremely powerful interface of defining exactly
what kinds of input data you support and how it should be validated. Unlike other LLM libraries, the definition
of strict types in code is how you navigate the complexity of working with stochastic text in your code.

"If the parsing succeeds, I'm now safe to use this data inside my code."

## Fundamentals

Every model in rigging should extend the [`Model`][rigging.model.Model] base class. This is a lightweight wrapper around pydantic-xml's [`BaseXMLModel`](`https://pydantic-xml.readthedocs.io/en/latest/pages/api.html#pydantic_xml.BaseXmlModel`)
with some added features and functionality to make it easy for Rigging to manage. In general this includes:

1. More intelligent parsing for the imperfect text which LLMs frequently provide. Nested tags, unclear sub-structures,
   multiple models scattered in the text, etc. See the [`.from_text()`][rigging.model.Model.from_text] method for the details.
2. A nicer [`.to_pretty_xml()`][rigging.model.Model.to_pretty_xml] to get new-line formatted outputs.
3. Some basic handling for escaping interior XML tags which normally wouldn't parse correctly.
4. Helpers to ensure the tag names from models are consistent and automatic.

However, everything pydantic-xml (and by extention normal pydantic) models support is also supported in Rigging.

!!! tip "Background Knowledge"

    If you happen to be new to modern python like type hinting and pydantic models, we would highly
    recommend you spend some time in their documentation to get more familiar. Without this background,
    many of the Rigging features will seem confusing.

    - [Python Typing](https://docs.python.org/3/library/typing.html)
    - [Pydantic](https://docs.pydantic.dev/)
    - [Pydantic XML](https://pydantic-xml.readthedocs.io/)

## Primitive Models

Often, you just want to indicate to the LLM that it should place a block of text between
tags so you can extract just that portion of the message content.

```py
import rigging as rg

class FunFact(rg.Model):
    fact: str
```

Pydantic XML refers to these as "primitive" because they have a single field. These models
leverage the minimal functionality of XML parsing and just take the contents between the start
and end tags.

??? note "Parsing for Primitive Models"

    We have a pending TODO to replace the internals of Pydantic XML parsing to make it
    more flexible for the kinds of "broken" XML that LLMs frequently produce.
    Primitive models have special handling in Rigging to make them more flexible
    for parsing this "broken" XML. If you are having issues with complex parsing,
    using primitive models is a good escape hatch for now.  

```py
import rigging as rg

class FunFact(rg.Model):
    fact: str

FunFact(fact="Rigging is pretty easy to use").to_pretty_xml()
# '<fun-fact>Rigging is pretty easy to use</fun-fact>'
```

Notice that the name of our interior field (`.fact`) isn't relevant to the XML structure. We only
use this to access the contents of that model field in code. However the name of our model (`FunFact`)
**is relevant** to the XML representation. What you name your models is important to how the underlying
LLM interprets it's meaning. You should be thoughtful and descriptive about your model names as they
will have effects on how well the LLM understands the intention, and how likely it is to output one
model over another (based on it's token probability distribution).

If you want to seperate the model tag from it's class name, you specify it in the class construction:

```py
class FunFact(rg.Model, tag="a-super-fun-fact")
    ...
```

## Typing and Validation

Even if models are primitive, we can still use type hinting and pydantic validation to ensure
that the content between tags conforms to any constraints we need. Take this example from a default
Rigging model for instance:

```py
class YesNoAnswer(Model):
    "Yes/No answer answer with coercion"

    boolean: bool
    """The boolean value of the answer."""

    @field_validator("boolean", mode="before")
    def parse_str_to_bool(cls, v: t.Any) -> t.Any:
        if isinstance(v, str):
            if v.strip().lower().startswith("yes"):
                return True
            elif v.strip().lower().startswith("no"):
                return False
        return v
```

You can see the interior field of the model is now a `bool` type, which means pydantic will accept standard
values which could be reasonably interpreted as a boolean. We also add a custom field validator to
check for instances of `yes/no` as text strings. All of these XML values will parse correctly:

```xml
<yes-no-answer>true</yes-no-answer>
<yes-no-answer>False</yes-no-answer>
<yes-no-answer>yes, it is.</yes-no-answer>
<yes-no-answer> NO </yes-no-answer>
<yes-no-answer>1</yes-no-answer>
```

The choice to build on Pydantic offers an incredible amount of flexibility for controlling exactly
how data is validated in your models. This kind of parsing work is exactly what these libraries were designed
to do. The sky is the limit, and **everything you find in Pydantic and Pydantic XML are compatible
with Rigging.**

## Handling Multiple Fields

Unlike vanilla Pydantic, our use of Pydantic XML forces us to think about exactly how models with multiple fields
will be represented in XML syntax. Take this as an example:

```py
class Person(rg.Model):
    name: str
    age: int
```

In XML this could be any of the following:

```xml
<person name="Will" age=30 />

<person name="Will">
    <age>30</age>
</person>

<person>
    <name>Will</name>
    <age>30</age>
</person>

...
```

*You get the idea.* Pydantic XML handles this all very well and offers different ways of defining
your fields to specific whether they should be **attributes** or child **elements**. You can read
more about this in [their documentation.](https://pydantic-xml.readthedocs.io/en/latest/pages/quickstart.html#primitives)

The basic rule is this: **If your model has more than one field, you need to define every field as
either an attribute or an element**

How exactly you structure your models and their associated representations is completely up to you.
Our general guide is that LLMs tend to work better with elements over attributes.


=== "Model Definition"

    ```py
    class Person(rg.Model):
        name: str = rg.element()
        age: int = rg.element()
    ```

=== "XML Format"

    ```xml
    <person>
        <name />
        <age />
    </person>
    ```

## XML Examples

For primitive models, using the default [`.xml_tags()`][rigging.model.Model.xml_tags] or [`.xml_example()`][rigging.model.Model.xml_example]
works well for communicating to the model how it should respond, however for more complex models it's **highly recommended** to overload
the [`.xml_example()`][rigging.model.Model.xml_example] method to provide a more detailed example of the XML structure you expect.

The easiest way to approach this overload is to instantiate your model class with some standard values
and use [`.to_prety_xml()`][rigging.model.Model.to_pretty_xml]


```py


## Complex Models

Let's design a model which will hold required information for making a web request. We'll begin with an outline of our model:

```py
class Header(rg.Model):
    name: str
    value: str

class Parameter(rg.Model):
    name: str
    value: str

class Request(rg.Model):
    method: str
    path: str
    headers: list
    url_params: list
    body: str
```

We'll start with a few standard string constraints to strip extra white-space (which LLMs tend to include)
and automatically convert our method to upper case.

```py
from pydantic import StringConstraints

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]
str_upper = t.Annotated[str, StringConstraints(to_upper=True)]

class Header(rg.Model):
    name: str
    value: str_strip

class Parameter(rg.Model):
    name: str
    value: str_strip

class Request(rg.Model):
    method: str_upper
    path: str_strip
    headers: list
    url_params: list
    body: str_strip
```

Next we'll assign our fields to attributes and elements.

```py
from pydantic import StringConstraints

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]
str_upper = t.Annotated[str, StringConstraints(to_upper=True)]

class Header(rg.Model):
    name: str = rg.attr()
    value: str_strip = rg.element()

class Parameter(rg.Model):
    name: str = rg.attr()
    value: str_strip = rg.element()

class Request(rg.Model):
    method: str_upper = rg.attr()
    path: str_strip = rg.attr()
    headers: list
    url_params: list
    body: str_strip = rg.element()
```

In terms of handling our headers and URL parameters, we want these to be a list of child
elements which are wrapped in a parent tag. We also want these and our body to be optional.

```py
from pydantic import StringConstraints

str_strip = t.Annotated[str, StringConstraints(strip_whitespace=True)]
str_upper = t.Annotated[str, StringConstraints(to_upper=True)]

class Header(rg.Model):
    name: str = rg.attr()
    value: str_strip

class Parameter(rg.Model):
    name: str = rg.attr()
    value: str_strip

class Request(rg.Model):
    method: str_upper = rg.attr()
    path: str = rg.attr()
    headers: list[Header] = rg.wrapped("headers", rg.element(default=[]))
    url_params: list[Parameter] = rg.wrapped("url-params", rg.element(default=[]))
    body: str_strip = rg.element(default="")
```

Let's check our final work:

=== "Model in Code"

    ```py
    Request(
        method="POST",
        path="/api/v1/search",
        headers=[
            Header(name="Authorization", value="Bearer sk-1234")
        ],
        url_params=[
            Parameter(name="max", value="100")
        ],
        body="search=rigging"
    )
    ```

=== "Model as XML"

    ```xml
    <request method="POST" path="/api/v1/search">
        <headers>
            <header name="Authorization">Bearer sk-1234</header>
        </headers>
        <url-params>
            <parameter name="max">100</parameter>
        </url-params>
        <body>search=rigging</body>
    </request>
    ```