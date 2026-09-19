from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal
from unittest.mock import Mock

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.fal_ai.cost_calculator import cost_calculator
from litellm.llms.fal_ai.image_generation import get_fal_ai_image_generation_config
from litellm.proxy.route_llm_request import route_request
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


def _background_removal_response(request: httpx.Request) -> httpx.Response:
    assert request.method == "POST"
    assert str(request.url) == f"https://fal.run/{MODEL}"
    body: Final = TypeAdapter(Mapping[str, JsonValue]).validate_json(request.content)
    assert body.get("image_url") == "https://example.com/input.png"
    assert body.get("sync_mode") is True
    assert "prompt" not in body
    return httpx.Response(200, content=b'{"image":{"url":"https://example.com/output.png"}}')


@pytest.mark.parametrize("use_router", (False, True))
@pytest.mark.parametrize("prompt_form", ("omitted", "null", "positional_null", "positional_string", "empty"))
def test_background_removal_sync_entry_points(
    use_router: bool,
    prompt_form: Literal["omitted", "null", "positional_null", "positional_string", "empty"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback: Final = Mock(spec=CustomLogger)
    monkeypatch.setattr(litellm, "input_callback", [callback])  # mutable-ok: LiteLLM callback registry requires a list.
    router: Final = litellm.Router(
        model_list=[  # mutable-ok: Router's public model_list contract requires a list of dictionaries.
            {  # mutable-ok: Router consumes mutable deployment dictionaries.
                "model_name": "remove-background",
                "litellm_params": {  # mutable-ok: Router reads and augments provider parameters.
                    "model": f"fal_ai/{MODEL}", "api_key": "test-key",
                },
            }
        ],
        num_retries=0,
    )
    with httpx.Client(transport=httpx.MockTransport(_background_removal_response)) as transport:
        client: Final = HTTPHandler(client=transport)
        generate: Final = router.image_generation if use_router else litellm.image_generation
        model: Final = "remove-background" if use_router else f"fal_ai/{MODEL}"
        response: Final = (
            generate(
                "ignored positional prompt" if prompt_form == "positional_string" else None,
                model,
                image_url="https://example.com/input.png",
                sync_mode=True,
                api_key="test-key",
                client=client,
            )
            if prompt_form in ("positional_null", "positional_string")
            else generate(
                model=model,
                image_url="https://example.com/input.png",
                sync_mode=True,
                api_key="test-key",
                client=client,
                **MappingProxyType(
                    {"prompt": None} if prompt_form == "null" else {"prompt": ""} if prompt_form == "empty" else {}
                ),
            )
        )
    callback.log_pre_api_call.assert_called_once()
    assert callback.log_pre_api_call.call_args.kwargs["messages"] == [
        {"role": "user", "content": "ignored positional prompt" if prompt_form == "positional_string" else ""}
    ]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert tuple(item.url for item in response.data) == ("https://example.com/output.png",)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_router", (False, True))
@pytest.mark.parametrize("explicit_null", (False, True))
async def test_promptless_proxy_dispatch_through_sdk_and_router(
    use_router: bool, explicit_null: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    callback: Final = Mock(spec=CustomLogger)
    monkeypatch.setattr(litellm, "input_callback", [callback])  # mutable-ok: LiteLLM callback registry requires a list.
    router: Final = litellm.Router(
        model_list=[  # mutable-ok: Router's public model_list contract requires a list of dictionaries.
            {  # mutable-ok: Router consumes mutable deployment dictionaries.
                "model_name": "remove-background",
                "litellm_params": {  # mutable-ok: Router reads and augments provider parameters.
                    "model": f"fal_ai/{MODEL}", "api_key": "test-key",
                },
            }
        ],
        num_retries=0,
    )
    with httpx.Client(transport=httpx.MockTransport(_background_removal_response)) as transport:
        client: Final = HTTPHandler(client=transport)
        call: Final = await route_request(
            data={  # mutable-ok: The proxy dispatcher augments the request dictionary in place.
                "model": "remove-background" if use_router else f"fal_ai/{MODEL}",
                "image_url": "https://example.com/input.png",
                "sync_mode": True,
                "client": client,
                **MappingProxyType({"prompt": None} if explicit_null else {}),
                **({} if use_router else {"api_key": "test-key"}),  # mutable-ok: Direct SDK dispatch needs credentials.
            },
            llm_router=router if use_router else None,
            user_model=None,
            route_type="aimage_generation",
        )
        response: Final = await call
    callback.log_pre_api_call.assert_called_once()
    assert callback.log_pre_api_call.call_args.kwargs["messages"] == [{"role": "user", "content": ""}]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert tuple(item.url for item in response.data) == ("https://example.com/output.png",)


@pytest.mark.parametrize("model", ("openai/gpt-image-2", "fal_ai/bria/text-to-image/3.2"))
@pytest.mark.parametrize("prompt_form", ("omitted", "null", "positional_null"))
def test_other_image_models_still_require_prompt(
    model: str, prompt_form: Literal["omitted", "null", "positional_null"],
) -> None:
    with pytest.raises(litellm.BadRequestError, match="requires a prompt"):
        if prompt_form == "positional_null":
            litellm.image_generation(None, model, api_key="test-key")
        else:
            litellm.image_generation(
                model=model, api_key="test-key",
                **MappingProxyType({"prompt": None} if prompt_form == "null" else {}),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("openai/gpt-image-2", "fal_ai/bria/text-to-image/3.2"))
@pytest.mark.parametrize("use_router", (False, True))
async def test_proxy_dispatch_rejects_promptless_generation_models(model: str, use_router: bool) -> None:
    router: Final = litellm.Router(
        model_list=[  # mutable-ok: Router's public model_list contract requires a list of dictionaries.
            {  # mutable-ok: Router consumes mutable deployment dictionaries.
                "model_name": "requires-prompt",
                "litellm_params": {"model": model, "api_key": "test-key"},  # mutable-ok: Router provider parameters.
            }
        ],
        num_retries=0,
    )
    call: Final = await route_request(
        data={  # mutable-ok: The proxy dispatcher augments the request dictionary in place.
            "model": "requires-prompt" if use_router else model,
            **({} if use_router else {"api_key": "test-key"}),  # mutable-ok: Direct SDK dispatch needs credentials.
        },
        llm_router=router if use_router else None,
        user_model=None,
        route_type="aimage_generation",
    )
    with pytest.raises(litellm.BadRequestError, match="requires a prompt"):
        await call


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
