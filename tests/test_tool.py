import inspect
import json
import typing as t
from dataclasses import dataclass
from textwrap import dedent

import pytest
from pydantic import BaseModel

import rigging as rg
from rigging.error import ToolDefinitionError
from rigging.model import Model, make_from_schema, make_from_signature
from rigging.tool.api import ApiFunctionCall, ApiFunctionDefinition, ApiToolCall, ApiToolDefinition
from rigging.tool.base import Tool
from rigging.tool.native import JsonInXmlToolCall, XmlToolCall, XmlToolDefinition

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001, FBT001, FBT002


def test_tool_from_simple_callable() -> None:
    """Test creating a tool from a simple callable."""

    def simple_function(name: str, age: int) -> str:
        """A simple function that returns a greeting."""
        return f"Hello {name}, you are {age} years old!"

    tool = Tool.from_callable(simple_function)

    assert tool.name == "simple_function"
    assert tool.description == "A simple function that returns a greeting."
    assert "_signature" in tool.__dict__
    assert "_type_adapter" in tool.__dict__

    # Check schema
    assert "name" in tool.parameters_schema["properties"]
    assert tool.parameters_schema["properties"]["name"]["type"] == "string"
    assert "age" in tool.parameters_schema["properties"]
    assert tool.parameters_schema["properties"]["age"]["type"] == "integer"


def test_tool_decorator() -> None:
    """Test the @tool decorator functionality."""

    @rg.tool
    def sample_tool(query: str, limit: int = 10) -> list[str]:
        """Search for items matching the query."""
        return [f"{query}-{i}" for i in range(limit)]

    assert isinstance(sample_tool, Tool)
    assert sample_tool.name == "sample_tool"
    assert "Search for items matching the query." in sample_tool.description

    # Test with custom name and description
    @rg.tool(name="custom_name", description="Custom description")
    def another_tool(x: int) -> int:
        return x * 2

    assert isinstance(another_tool, Tool)
    assert another_tool.name == "custom_name"
    assert another_tool.description == "Custom description"


def test_api_definition_generation() -> None:
    """Test that tools correctly generate API definitions."""

    def complex_function(
        name: str,
        age: int,
        tags: list[str] = ["default"],  # noqa: B006
        active: bool = True,
    ) -> dict[str, t.Any]:
        """Process user data with complex parameters."""
        return {"name": name, "age": age, "tags": tags, "active": active}

    tool = Tool.from_callable(complex_function)
    api_def = tool.api_definition

    assert isinstance(api_def, ApiToolDefinition)
    assert api_def.type == "function"
    assert isinstance(api_def.function, ApiFunctionDefinition)
    assert api_def.function.name == "complex_function"
    assert api_def.function.description is not None
    assert "Process user data with complex parameters." in api_def.function.description

    # Check parameters schema
    params = api_def.function.parameters
    assert isinstance(params, dict)
    assert params["type"] == "object"
    assert "name" in params["properties"]
    assert "age" in params["properties"]
    assert "tags" in params["properties"]
    assert "active" in params["properties"]

    # Check required parameters
    assert "name" in params["required"]
    assert "age" in params["required"]
    assert "tags" not in params["required"]
    assert "active" not in params["required"]


def test_xml_definition_generation() -> None:
    """Test that tools correctly generate XML definitions."""

    def profile_function(user_id: str, include_details: bool = False) -> dict[str, t.Any]:
        """Get user profile information."""
        return {"id": user_id, "details": include_details}

    tool = Tool.from_callable(profile_function)
    xml_def = tool.xml_definition

    assert isinstance(xml_def, XmlToolDefinition)
    assert xml_def.name == "profile_function"
    assert "Get user profile information." in xml_def.description

    # XML parameters should contain both params
    assert '<param name="user_id"' in xml_def.parameters
    assert '<param name="include_details"' in xml_def.parameters

    # Required param should be marked as such
    assert 'required="true"' in xml_def.parameters
    # Optional param should not be marked as required
    assert 'required="false"' in xml_def.parameters


def test_json_definition_generation() -> None:
    """Test that tools correctly generate JSON-in-XML definitions."""

    def data_function(query: str, max_results: int = 10) -> list[str]:
        """Query data with pagination."""
        return [f"result-{i}" for i in range(max_results)]

    tool = Tool.from_callable(data_function)
    json_def = tool.json_definition

    assert json_def.name == "data_function"
    assert "Query data with pagination." in json_def.description

    # Parameters should be JSON schema
    params = json.loads(json_def.parameters)
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert "max_results" in params["properties"]
    assert "query" in params["required"]


def test_annotated_parameter_descriptions() -> None:
    """Test that Annotated types with descriptions are properly handled."""

    def annotated_function(
        simple: str,
        described: t.Annotated[int, "Number of items to process"],
        optional: t.Annotated[bool, "Enable feature flag"] = False,
    ) -> str:
        """Function with annotated parameters."""
        return f"{simple} {described} {optional}"

    tool = Tool.from_callable(annotated_function)

    # Check schema descriptions
    schema = tool.parameters_schema
    assert schema["properties"]["simple"].get("description") is None
    assert schema["properties"]["described"]["description"] == "Number of items to process"
    assert schema["properties"]["optional"]["description"] == "Enable feature flag"

    # Check API definition
    api_def = tool.api_definition
    api_params = api_def.function.parameters
    assert api_params["properties"]["described"]["description"] == "Number of items to process"  # type: ignore [index]

    # Check XML definition
    xml_def = tool.xml_definition
    assert "Number of items to process" in xml_def.parameters
    assert "Enable feature flag" in xml_def.parameters


def test_tool_model_creation() -> None:
    """Test that the tool correctly creates a Model for XML parsing."""

    def config_function(name: str, version: int, features: list[str] = []) -> dict[str, t.Any]:  # noqa: B006
        """Configure application settings."""
        return {"name": name, "version": version, "features": features}

    tool = Tool.from_callable(config_function)

    # Access model property to create the model
    model = tool.model

    # Verify model is properly created
    assert issubclass(model, Model)
    assert hasattr(model, "model_fields")
    assert "name" in model.model_fields
    assert "version" in model.model_fields
    assert "features" in model.model_fields


class TestToolHandleCall:
    """Test suite for tool call handling."""

    @pytest.fixture
    def sample_tool(self) -> Tool[..., t.Any]:
        def calculator(a: int, b: int, operation: str = "add") -> int:
            """Perform math operations."""
            if operation == "add":
                return a + b
            if operation == "multiply":
                return a * b
            if operation == "subtract":
                return a - b
            raise ValueError(f"Unknown operation: {operation}")

        return Tool.from_callable(calculator)

    @pytest.mark.asyncio
    async def test_handle_api_tool_call(self, sample_tool: Tool[..., t.Any]) -> None:
        """Test handling API format tool calls."""
        from rigging.tool.api import ApiFunctionCall, ApiToolCall

        tool_call = ApiToolCall(
            id="call123",
            function=ApiFunctionCall(
                name="calculator",
                arguments=json.dumps({"a": 5, "b": 3, "operation": "multiply"}),
            ),
        )

        message, stop = await sample_tool.handle_tool_call(tool_call)

        assert stop is False
        assert message is not None
        assert message.role == "tool"
        assert message.tool_call_id == "call123"
        assert message.content == "15"

    @pytest.mark.asyncio
    async def test_handle_xml_tool_call(self, sample_tool: Tool[..., t.Any]) -> None:
        """Test handling XML format tool calls."""
        tool_call = XmlToolCall(
            name="calculator",
            parameters=dedent(
                """
                <a>10</a>
                <b>2</b>
                <operation>subtract</operation>
            """,
            ).strip(),
        )

        message, stop = await sample_tool.handle_tool_call(tool_call)

        assert stop is False
        assert message is not None
        assert message.role == "user"
        assert message.content == '<rg:tool-result name="calculator">8</rg:tool-result>'

    @pytest.mark.asyncio
    async def test_handle_json_xml_tool_call(self, sample_tool: Tool[..., t.Any]) -> None:
        """Test handling JSON-in-XML format tool calls."""
        tool_call = JsonInXmlToolCall(
            name="calculator",
            parameters=json.dumps({"a": 4, "b": 4, "operation": "add"}),
        )

        message, stop = await sample_tool.handle_tool_call(tool_call)

        assert stop is False
        assert message is not None
        assert message.role == "user"
        assert message.content == '<rg:tool-result name="calculator">8</rg:tool-result>'


def test_make_from_signature() -> None:
    """Test the make_from_signature function directly."""

    def test_func(name: str, age: int, tags: list[str] = []) -> None:  # noqa: B006
        """Test function for signature extraction."""

    signature = inspect.signature(test_func)
    model_class = make_from_signature(signature, "TestParams")

    assert issubclass(model_class, Model)
    assert "name" in model_class.model_fields
    assert "age" in model_class.model_fields
    assert "tags" in model_class.model_fields


def test_make_from_schema() -> None:
    """Test the make_from_schema function directly."""
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Max results to return"},
            "filters": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    }

    model_class = make_from_schema(schema, "SearchParams")

    assert issubclass(model_class, Model)
    assert "query" in model_class.model_fields
    assert "limit" in model_class.model_fields
    assert "filters" in model_class.model_fields

    # Check that description was preserved
    assert model_class.model_fields["limit"].description == "Max results to return"


def test_prompt_integration() -> None:
    """Test integration between Tool and Prompt objects."""

    @rg.prompt
    async def generate_greeting(name: str, formal: bool = False) -> str:  # type: ignore [empty-body]
        """Generate a greeting for the user."""

    tool = Tool.from_callable(generate_greeting)

    assert tool.name == "generate_greeting"
    assert "Generate a greeting for the user." in tool.description

    # Verify parameters
    schema = tool.parameters_schema
    assert "name" in schema["properties"]
    assert "formal" in schema["properties"]
    assert schema["properties"]["formal"]["default"] is False


def test_complex_model_parameters() -> None:
    """Test tools with complex model parameters."""

    class UserProfile(BaseModel):
        name: str
        age: int

    @dataclass
    class UserSettings:
        notifications: bool
        theme: str

    def process_user(profile: UserProfile, update: bool = False) -> dict[str, t.Any]:
        """Process a user profile."""
        return {"profile": profile, "updated": update}

    def process_settings(settings: UserSettings) -> dict[str, t.Any]:
        """Process user settings."""
        return {"settings": settings}

    # This should raise an error since pydantic models should be BaseXmlModel
    with pytest.raises(ToolDefinitionError):
        Tool.from_callable(process_user).xml_definition  # noqa: B018

    # This should raise an error since dataclasses aren't supported
    with pytest.raises(ToolDefinitionError):
        Tool.from_callable(process_settings).xml_definition  # noqa: B018


@pytest.mark.asyncio
async def test_tool_error_catching() -> None:
    """Test that errors in tool functions are caught and reported."""

    def faulty_function(x: int) -> int:
        """A function that raises an error."""
        raise ValueError("This is a test error")

    tool = Tool.from_callable(faulty_function)
    tool_call = ApiToolCall(
        id="call123",
        function=ApiFunctionCall(name="faulty_function", arguments='{"x": 5}'),
    )

    with pytest.raises(ValueError, match="This is a test error"):
        await tool.handle_tool_call(tool_call)

    tool = Tool.from_callable(faulty_function, catch={RuntimeError})

    with pytest.raises(ValueError, match="This is a test error"):
        await tool.handle_tool_call(tool_call)

    tool = Tool.from_callable(faulty_function, catch={ValueError})

    message, stop = await tool.handle_tool_call(tool_call)

    assert stop is False
    assert "This is a test error" in message.content
