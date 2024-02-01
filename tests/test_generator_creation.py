import pytest

from rigging.error import InvalidModelSpecifiedError
from rigging.generator import GenerateParams, LiteLLMGenerator, get_generator


@pytest.mark.parametrize("identifier", ["test_model", "litellm!test_model"])
def test_get_generator_default_is_litellm(identifier: str) -> None:
    generator = get_generator(identifier)
    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "test_model"


@pytest.mark.parametrize("identifier", ["invalid!testing", "no_exist!stuff,args=123"])
def test_get_generator_invalid_provider(identifier: str) -> None:
    with pytest.raises(InvalidModelSpecifiedError):
        get_generator(identifier)


@pytest.mark.parametrize(
    "identifier, valid_params",
    [
        ("litellm!test_model,max_tokens=123,top_p=10", GenerateParams(max_tokens=123, top_p=10)),
        ("litellm!test_model,temperature=0.5", GenerateParams(temperature=0.5)),
        ("test_model,max_tokens=100,temperature=1.0", GenerateParams(max_tokens=100, temperature=1.0)),
    ],
)
def test_get_generator_with_params(identifier: str, valid_params: GenerateParams) -> None:
    generator = get_generator(identifier)
    assert isinstance(generator, LiteLLMGenerator)
    assert generator.model == "test_model"
    assert generator.params == valid_params


@pytest.mark.parametrize("identifier", ["litellm:invalid,stuff:test,t1/123", "litellm:invalid,stuff:test,t1/123"])
def test_get_generator_invalid_structure_format(identifier: str) -> None:
    with pytest.raises(InvalidModelSpecifiedError):
        get_generator(identifier)


@pytest.mark.parametrize(
    "identifier", ["litellm:model,bad_param=123,temperature=1.0", "litellm:model,temperature=True"]
)
def test_get_generator_invalid_params(identifier: str) -> None:
    with pytest.raises(InvalidModelSpecifiedError):
        get_generator(identifier)
