import typing as t
from textwrap import dedent
from xml.sax.saxutils import escape

import pytest

from rigging.model import Model, attr, element

# mypy: disable-error-code=empty-body
# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001


class SimpleModel(Model):
    """A simple model to test basic element generation."""

    content: str = element(examples=["Hello, World!"])
    """The main content of the model."""


class NoExampleModel(Model):
    """A model to test fallback to an empty element when no example is given."""

    name: str
    """The name of the entity."""


class AttrAndElementModel(Model):
    """Tests a model with both an attribute and a child element."""

    id: int = attr(examples=[123])
    """The unique identifier (attribute)."""
    value: str = element(examples=["Some value"])
    """The primary value (element)."""


class DocstringDescriptionModel(Model):
    """Tests that field docstrings are correctly used as descriptions."""

    field1: str = element(examples=["val1"])
    """This is the description for field1."""
    field2: bool = element(examples=[True])
    """This is the description for field2."""


class ParameterDescriptionModel(Model):
    """Tests that the `description` parameter overrides a field's docstring."""

    param: str = element(
        examples=["override"], description="This description is from the `description` parameter."
    )
    """This docstring should be ignored in the XML example."""


class SpecialCharsModel(Model):
    """Tests proper escaping of special XML characters in examples and comments."""

    comment: str = element(examples=["ok"])
    """This comment contains < and > & special characters."""
    data: str = element(examples=["<tag>&'"])
    """This element's example contains special XML characters."""


# This class definition is based on the one you provided in the prompt.
class Analysis(Model, tag="analysis"):
    """A model to validate the exact output requested in the prompt."""

    priority: t.Literal["low", "medium", "high", "critical"] = element(examples=["medium"])
    """Triage priority for human follow-up."""
    tags: str = element("tags", examples=["admin panel, error message, legacy"])
    """A list of specific areas within the screenshot that are noteworthy or require further examination."""
    summary: str = element()
    """A markdown summary explaining *why* the screenshot is interesting and what a human should investigate next."""


@pytest.mark.parametrize(
    ("model_cls", "expected_xml"),
    [
        pytest.param(
            SimpleModel,
            """
            <simple-model>
              <!-- The main content of the model. -->
              <content>Hello, World!</content>
            </simple-model>
            """,
            id="simple_model",
        ),
        pytest.param(
            NoExampleModel,
            """
            <no-example-model></no-example-model>
            """,
            id="model_with_no_example",
        ),
        pytest.param(
            AttrAndElementModel,
            """
            <attr-and-element-model id="123">
              <!-- The primary value (element). -->
              <value>Some value</value>
            </attr-and-element-model>
            """,
            id="model_with_attribute_and_element",
        ),
        pytest.param(
            DocstringDescriptionModel,
            """
            <docstring-description-model>
              <!-- This is the description for field1. -->
              <field1>val1</field1>
              <!-- This is the description for field2. -->
              <field2>True</field2>
            </docstring-description-model>
            """,
            id="descriptions_from_docstrings",
        ),
        pytest.param(
            ParameterDescriptionModel,
            """
            <parameter-description-model>
              <!-- This description is from the `description` parameter. -->
              <param>override</param>
            </parameter-description-model>
            """,
            id="description_from_parameter_overrides_docstring",
        ),
        pytest.param(
            SpecialCharsModel,
            f"""
            <special-chars-model>
              <!-- {escape("This comment contains < and > & special characters.")} -->
              <comment>ok</comment>
              <!-- {escape("This element's example contains special XML characters.")} -->
              <data>{escape("<tag>&'")}</data>
            </special-chars-model>
            """,
            id="escaping_of_special_characters",
        ),
        pytest.param(
            Analysis,
            """
            <analysis>
              <!-- Triage priority for human follow-up. -->
              <priority>medium</priority>
              <!-- A list of specific areas within the screenshot that are noteworthy or require further examination. -->
              <tags>admin panel, error message, legacy</tags>
              <!-- A markdown summary explaining *why* the screenshot is interesting and what a human should investigate next. -->
              <summary/>
            </analysis>
            """,
            id="user_provided_analysis_model",
        ),
    ],
)
def test_xml_example_generation(model_cls: type[Model], expected_xml: str) -> None:
    """
    Validates that the `xml_example()` class method produces the correct
    pretty-printed XML with examples and descriptions as comments.
    """
    actual_xml = model_cls.xml_example()
    assert dedent(actual_xml).strip() == dedent(expected_xml).strip()
