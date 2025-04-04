from dataclasses import dataclass
from textwrap import dedent
from typing import Annotated

import pytest

import rigging as rg
from rigging.chat import Chat

# mypy: disable-error-code=empty-body
# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001


def test_prompt_render_docstring_parse() -> None:
    @rg.prompt
    async def foo(name: str) -> str:
        """Say hello."""
        ...

    assert foo.docstring == "Say hello."

    @rg.prompt
    async def bar(name: str) -> str:
        """
        Say hello."""
        ...

    assert bar.docstring == "Say hello."

    @rg.prompt
    async def baz(name: str) -> str:
        """
        Say \
        hello.

        """
        ...

    assert baz.docstring == "Say hello."


def test_basic_prompt_render() -> None:
    @rg.prompt
    async def hello(name: str) -> str:
        """Say hello."""
        ...

    rendered = hello.render("Alice")
    assert rendered == dedent(
        """\
    Say hello.

    <name>Alice</name>

    Produce the following output (use xml tags):

    <str></str>
    """,
    )


def test_prompt_render_with_docstring_variables() -> None:
    @rg.prompt
    async def greet(name: str, greeting: str = "Hello") -> str:
        """Say '{{ greeting }}' to {{ name }}."""
        ...

    rendered = greet.render("Bob")
    assert rendered == dedent(
        """\
    Say 'Hello' to Bob.

    Produce the following output (use xml tags):

    <str></str>
    """,
    )


def test_prompt_render_with_model_output() -> None:
    class Person(rg.Model):
        name: str = rg.element()
        age: int = rg.element()

    @rg.prompt
    async def create_person(name: str, age: int) -> Person:
        """Create a person."""
        ...

    rendered = create_person.render("Alice", 30)
    assert rendered == dedent(
        """\
    Create a person.

    <name>Alice</name>

    <age>30</age>

    Produce the following output (use xml tags):

    <person>
      <name/>
      <age/>
    </person>
    """,
    )


def test_prompt_render_with_list_output() -> None:
    @rg.prompt
    async def generate_numbers(count: int) -> list[int]:
        """Generate a list of numbers."""
        ...

    rendered = generate_numbers.render(5)
    assert rendered == dedent(
        """\
    Generate a list of numbers.

    <count>5</count>

    Produce the following output for each item (use xml tags):

    <int></int>
    """,
    )


def test_prompt_render_with_tuple_output() -> None:
    @rg.prompt
    async def create_user(username: str) -> tuple[str, int]:
        """Create a new user."""
        ...

    rendered = create_user.render("johndoe")
    assert rendered == dedent(
        """\
    Create a new user.

    <username>johndoe</username>

    Produce the following outputs (use xml tags):

    <str></str>

    <int></int>
    """,
    )


def test_prompt_render_with_tuple_output_ctx() -> None:
    @rg.prompt
    async def create_user(username: str) -> tuple[Annotated[str, rg.Ctx(tag="id")], int]:
        """Create a new user."""
        ...

    rendered = create_user.render("johndoe")
    assert rendered == dedent(
        """\
    Create a new user.

    <username>johndoe</username>

    Produce the following outputs (use xml tags):

    <id></id>

    <int></int>
    """,
    )


def test_prompt_render_with_dataclass_output() -> None:
    @dataclass
    class User:
        username: str
        email: str
        age: int

    @rg.prompt
    async def register_user(username: str, email: str, age: int) -> User:
        """Register a new user: {{ username}}."""
        ...

    rendered = register_user.render("johndoe", "johndoe@example.com", 25)
    assert rendered == dedent(
        """\
    Register a new user: johndoe.

    <email>johndoe@example.com</email>

    <age>25</age>

    Produce the following outputs (use xml tags):

    <username></username>

    <email></email>

    <age></age>
    """,
    )


def test_prompt_render_with_chat_return() -> None:
    @rg.prompt
    async def foo(input_: str) -> Chat:
        """Do something."""
        ...

    rendered = foo.render("bar")
    assert rendered == dedent(
        """\
    Do something.

    <input>bar</input>
    """,
    )


def test_prompt_render_ctx_in_dataclass() -> None:
    @dataclass
    class User:
        username: str
        email: Annotated[str, rg.Ctx(prefix="The user email:", example="[test@email.com]")]
        age: Annotated[int, rg.Ctx(tag="override")]

    @rg.prompt
    async def register_user(username: str, email: str, age: int) -> User:
        """Register a new user: {{ username }}."""
        ...

    rendered = register_user.render("johndoe", "john@email.com", 30)
    assert rendered == dedent(
        """\
    Register a new user: johndoe.

    <email>john@email.com</email>

    <age>30</age>

    Produce the following outputs (use xml tags):

    <username></username>

    The user email:
    <email>[test@email.com]</email>

    <override></override>
    """,
    )


def test_prompt_parse_fail_nested_input() -> None:
    async def foo(arg: list[list[str]]) -> Chat:
        ...

    with pytest.raises(TypeError):
        rg.prompt(foo)

    async def bar(arg: tuple[int, str, tuple[str]]) -> Chat:
        ...

    with pytest.raises(TypeError):
        rg.prompt(bar)


def test_prompt_parse_fail_unique_ouput() -> None:
    async def foo(arg: int) -> tuple[str, str]:
        ...

    with pytest.raises(TypeError):
        rg.prompt(foo)
