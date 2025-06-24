import pytest

from rigging.error import InvalidGeneratorError
from rigging.generator import (
    GenerateParams,
    LiteLLMGenerator,
    get_generator,
    get_identifier,
    register_generator,
)

from .generators import EchoGenerator

# ruff: noqa: S101, PLR2004, ARG001, PT011, SLF001


@pytest.mark.parametrize("identifier", ["test_model", "litellm!test_model"])
def test_get_generator_default_is_litellm(identifier: str) -> None:
    generator = get_generator(identifier)
    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "test_model"


@pytest.mark.parametrize("identifier", ["invalid!testing", "no_exist!stuff,args=123"])
def test_get_generator_invalid_provider(identifier: str) -> None:
    with pytest.raises(InvalidGeneratorError):
        get_generator(identifier)


@pytest.mark.parametrize(
    ("identifier", "valid_params"),
    [
        ("litellm!test_model,max_tokens=123,top_p=10", GenerateParams(max_tokens=123, top_p=10)),
        ("litellm!test_model,temperature=0.5", GenerateParams(temperature=0.5)),
        (
            "test_model,temperature=1.0,max_tokens=100",
            GenerateParams(max_tokens=100, temperature=1.0),
        ),
    ],
)
def test_get_generator_with_params(identifier: str, valid_params: GenerateParams) -> None:
    generator = get_generator(identifier)
    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "test_model"
    assert generator.params == valid_params


@pytest.mark.parametrize(
    "identifier",
    [
        ("litellm!test_model,max_tokens=1024,top_p=0.1"),
        ("litellm!custom,temperature=1.0,max_tokens=100,api_base=https://localhost:8000"),
        ("litellm!many/model/slashes,stop=a;b;c;"),
        ("litellm!with_cls_args,max_connections=10"),
    ],
)
def test_identifier_roundtrip(identifier: str) -> None:
    generator = get_generator(identifier)
    assert generator.to_identifier() == identifier


def test_get_identifier_no_extra() -> None:
    generator = get_generator("testing_model,temperature=0.5")
    generator.params.extra = {"abc": 123}
    identifier = get_identifier(generator)
    assert "extra" not in identifier


@pytest.mark.parametrize(
    "identifier",
    ["litellm:invalid,stuff:test,t1/123", "bad:invalid,stuff:test,t1//;;123:"],
)
def test_get_generator_invalid_structure_format(identifier: str) -> None:
    with pytest.raises(InvalidGeneratorError):
        get_generator(identifier)


@pytest.mark.parametrize(
    "identifier",
    ["litellm:model,bad_param=123,temperature=1.0", "litellm:model,temperature=True"],
)
def test_get_generator_invalid_params(identifier: str) -> None:
    with pytest.raises(InvalidGeneratorError):
        get_generator(identifier)


def test_register_generator() -> None:
    with pytest.raises(InvalidGeneratorError):
        get_generator("echo!test")

    register_generator("echo", EchoGenerator)
    generator = get_generator("echo!test")
    assert isinstance(generator, EchoGenerator)


def test_get_generator_b64() -> None:
    generator = get_generator("litellm!test_model,api_key=ZXhhbXBsZXRleHQ=")
    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "test_model"
