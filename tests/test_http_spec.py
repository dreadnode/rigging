import json
import pytest
from pydantic import ValidationError

from rigging.error import ProcessingError
from rigging.generator.http import HTTPSpec, RequestTransformContext


def get_basic_context() -> RequestTransformContext:
    return RequestTransformContext(
        role="user",
        content="Hello world",
        all_content="Hello world",
        messages=[{"role": "user", "content": "Hello world"}],
        params={"temperature": 0.7},
        api_key="test-key",
        model="test-model",
    )


def test_basic_json_transform() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/v1/chat",
            "method": "POST",
            "headers": {"Authorization": "Bearer {{api_key}}"},
            "transforms": [{"type": "json", "pattern": {"model": "$model", "messages": "$messages"}}],
        }
    )

    ctx = get_basic_context()
    body = spec.make_request_body(ctx)
    headers = spec.make_headers(ctx)
    url = spec.make_url(ctx)

    assert "test-model" in body
    assert "Hello world" in body
    assert headers["Authorization"] == "Bearer test-key"
    assert url == "https://api.example.com/v1/chat"


def test_complex_jinja_transform() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/{{model}}/generate",
            "method": "POST",
            "headers": {"Authorization": "Bearer {{api_key}}", "X-Model": "{{model}}"},
            "transforms": [
                {
                    "type": "jinja",
                    "pattern": """
                    {
                        "prompt": "{{content|tojson}}",
                        "system_params": {{params|tojson}},
                        "conversation": {{messages|tojson}},
                        "metadata": {
                            "model": "{{model}}",
                            "role": "{{role}}"
                        }
                    }
                    """,
                }
            ],
        }
    )

    ctx = get_basic_context()
    body = spec.make_request_body(ctx)
    headers = spec.make_headers(ctx)
    url = spec.make_url(ctx)

    assert "Hello world" in body
    assert '"temperature": 0.7' in body
    assert '"model": "test-model"' in body
    assert headers["X-Model"] == "test-model"
    assert url == "https://api.example.com/test-model/generate"


def test_chained_transforms() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/chat",
            "transforms": [
                {"type": "json", "pattern": {"raw_messages": "$messages"}},
                {
                    "type": "jinja",
                    "pattern": """
                    {
                        "formatted_messages": {{data.raw_messages|tojson}},
                        "additional_context": {
                            "model": "{{model}}",
                            "temperature": {{params.temperature}}
                        }
                    }
                    """,
                },
            ],
        }
    )

    ctx = get_basic_context()
    body = spec.make_request_body(ctx)

    assert "Hello world" in body
    assert "formatted_messages" in body
    assert "additional_context" in body
    assert "test-model" in body


def test_response_parsing() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/chat",
            "transforms": [{"type": "json", "pattern": {"prompt": "$content"}}],
        },
        response={
            "valid_status_codes": [200, 201],
            "transforms": [
                {"type": "jsonpath", "pattern": "$.choices[0].message.content"},
                {"type": "jinja", "pattern": "{{result|trim}}"},
            ],
        },
    )

    mock_response = """
    {
        "choices": [
            {
                "message": {
                    "content": "  Parsed response  "
                }
            }
        ]
    }
    """

    result = spec.parse_response_body(mock_response)
    assert result == "Parsed response"


def test_regex_response_transform() -> None:
    spec = HTTPSpec(
        request={"url": "https://api.example.com/chat", "transforms": [{"type": "json", "pattern": {}}]},
        response={"transforms": [{"type": "regex", "pattern": r"content:\s*'([^']*)'"}]},
    )

    mock_response = "Response received with content: 'Hello world'"
    result = spec.parse_response_body(mock_response)
    assert result == "Hello world"


def test_invalid_transform_type() -> None:
    with pytest.raises(ValidationError):
        spec = HTTPSpec(
            request={"url": "https://api.example.com/chat", "transforms": [{"type": "invalid", "pattern": {}}]}
        )
        ctx = get_basic_context()
        spec.make_request_body(ctx)


def test_missing_variable_in_template() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/chat",
            "transforms": [{"type": "jinja", "pattern": "{{missing_variable}}"}],
        }
    )

    ctx = get_basic_context()
    with pytest.raises(ProcessingError):
        spec.make_request_body(ctx)


def test_nested_json_transforms() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/chat",
            "transforms": [
                {
                    "type": "json",
                    "pattern": {
                        "conversation": {"messages": "$messages", "metadata": {"model": "$model", "params": "$params"}}
                    },
                }
            ],
        }
    )

    ctx = get_basic_context()
    body = spec.make_request_body(ctx)

    assert "Hello world" in body
    assert "test-model" in body
    assert "0.7" in body


def test_custom_header_templates() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/chat",
            "headers": {
                "Authorization": "Bearer {{api_key}}",
                "X-Request-ID": "{{params.request_id|default('default-id')}}",
                "X-Model-Version": "{{model}}-{{params.version|default('v1')}}",
            },
            "transforms": [{"type": "json", "pattern": {}}],
        }
    )

    ctx = get_basic_context()
    headers = spec.make_headers(ctx)

    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Request-ID"] == "default-id"
    assert headers["X-Model-Version"] == "test-model-v1"


def test_parse_response_body_int() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/v1/chat",
            "method": "POST",
            "headers": {"Authorization": "Bearer {{api_key}}"},
            "transforms": [{"type": "json", "pattern": {"model": "$model", "messages": "$messages"}}],
        },
        response={
            "valid_status_codes": [200, 201],
            "transforms": [{"type": "jsonpath", "pattern": "$.int_value"}],
        },
    )
    result = spec.parse_response_body('{"int_value": 42}')
    assert result == "42"


def test_parse_response_body_list() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/v1/chat",
            "method": "POST",
            "headers": {"Authorization": "Bearer {{api_key}}"},
            "transforms": [{"type": "json", "pattern": {"model": "$model", "messages": "$messages"}}],
        },
        response={
            "valid_status_codes": [200, 201],
            "transforms": [{"type": "jsonpath", "pattern": "foo[*].baz"}],
        },
    )
    result = spec.parse_response_body('{"foo": [{"baz": 1}, {"baz": 2}]}')
    assert result == "[1, 2]"


def test_parse_single_value_response_body() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/v1/chat",
            "method": "POST",
            "headers": {"Authorization": "Bearer {{api_key}}"},
            "transforms": [{"type": "json", "pattern": {"model": "$model", "messages": "$messages"}}],
        },
        response={
            "valid_status_codes": [200, 201],
            "transforms": [{"type": "jsonpath", "pattern": "$.foo"}],
        },
    )
    result = spec.parse_response_body('{"foo": 1, "bar": 2}')
    assert result == "1"



def test_jsonpath_transform_into_json() -> None:
    spec = HTTPSpec(
        request={
            "url": "https://api.example.com/v1/chat",
            "method": "POST",
            "headers": {"Authorization": "Bearer {{api_key}}"},
            "transforms": [{"type": "json", "pattern": {"model": "$model", "messages": "$messages"}}],
        },
        response={
            "valid_status_codes": [200, 201],
            "transforms": [{"type": "jsonpath", "pattern": "$"}],
        },
    )
    result = spec.parse_response_body('{"foo": [{"baz": 1}, {"baz": 2}]}')
    assert result
    assert json.loads(result)
