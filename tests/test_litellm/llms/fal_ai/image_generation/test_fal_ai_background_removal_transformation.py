from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.llms.fal_ai.image_generation import get_fal_ai_image_generation_config
from litellm.types.utils import ImageObject, ImageResponse

MODEL = "fal-ai/bria/background/remove"


@pytest.mark.parametrize("model", [MODEL, "bria/background/remove", f"fal_ai/{MODEL}", "fal_ai/bria/background/remove"])
def test_config_selection_and_url(model):
    config = get_fal_ai_image_generation_config(model)
    assert type(config).__name__ == "FalAIBackgroundRemovalConfig"
    assert config.get_complete_url(None, None, model, {}, {}) == f"https://fal.run/{MODEL}"


def test_bria_generation_keeps_its_config():
    assert type(get_fal_ai_image_generation_config("fal-ai/bria/text-to-image/base")).__name__ == "FalAIBriaConfig"


@pytest.mark.parametrize("sync_mode", [False, True])
def test_params_passthrough_and_unconditional_prompt_drop(sync_mode):
    config = get_fal_ai_image_generation_config(MODEL)
    ignored = {
        "response_format": "b64_json",
        "n": 2,
        "size": "1024x1024",
        "quality": "hd",
        "style": "vivid",
        "background": "opaque",
    }
    assert set(ignored) <= set(config.get_supported_openai_params(MODEL))
    expected = {"image_url": "https://example.com/input.png", "sync_mode": sync_mode}
    mapped = config.map_openai_params(
        {**ignored, **expected, "prompt": "ignored", "image_url": "https://example.com/other.png"},
        {**expected, **ignored, "prompt": "also ignored"},
        MODEL,
        False,
    )
    assert mapped == expected
    body = config.transform_image_generation_request(
        MODEL, "supplied prompt", {**mapped, **ignored, "prompt": "extra prompt"}, {}, {}
    )
    assert body == expected


def test_missing_image_url_raises():
    with pytest.raises(ValueError, match="image_url"):
        get_fal_ai_image_generation_config(MODEL).transform_image_generation_request(MODEL, "", {}, {}, {})


@pytest.mark.parametrize("field", ["image", "images"])
def test_response_single_object_and_array_fallback(field):
    image = {"url": "https://example.com/output.png", "content_type": "image/png", "width": 1024, "height": 1024}
    raw = httpx.Response(200, json={field: image if field == "image" else [image]})
    response = get_fal_ai_image_generation_config(MODEL).transform_image_generation_response(
        MODEL,
        raw,
        ImageResponse(),
        Mock(),
        {},
        {},
        {},
        None,
    )
    assert [item.url for item in response.data] == [image["url"]]


def test_flat_image_cost(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    response = ImageResponse(data=[ImageObject(url="https://example.com/output.png")])
    assert cost_calculator(MODEL, response) == pytest.approx(0.018)


def test_invalid_json_raises_provider_error():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    with pytest.raises(BaseLLMException, match="Error transforming"):
        get_fal_ai_image_generation_config(MODEL).transform_image_generation_response(
            MODEL,
            httpx.Response(200, text="not json"),
            ImageResponse(),
            Mock(),
            {},
            {},
            {},
            None,
        )
