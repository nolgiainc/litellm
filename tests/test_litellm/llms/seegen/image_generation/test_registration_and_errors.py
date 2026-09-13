import math
from pathlib import Path

import pytest
from pydantic import JsonValue as PydanticJsonValue
from pydantic import TypeAdapter

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.seegen.common_utils import JsonValue, SeeGenError, error_from_response
from litellm.llms.seegen.image_generation.transformation import SeeGenImageGenerationConfig
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails, LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize(
    ("status_code", "payload", "message"),
    [
        (402, {"error": "insufficient_balance", "message": "balance below threshold"}, "insufficient_balance"),
        (
            400,
            {
                "error": {
                    "code": "invalid_size",
                    "message": "bad size",
                    "param": "size",
                    "type": "invalid_request_error",
                }
            },
            "invalid_size",
        ),
    ],
)
def test_error_from_response_handles_both_vendor_shapes(
    status_code: int,
    payload: dict[str, JsonValue],
    message: str,
) -> None:
    response = error_from_response(status_code=status_code, payload=payload, headers={})

    assert isinstance(response, SeeGenError)
    assert response.status_code == status_code
    assert message in response.message


def test_provider_routing_config_and_pricing_are_registered() -> None:
    routed_model, provider, _, _ = get_llm_provider("seegen/seedream-v4.0")
    config = ProviderConfigManager.get_provider_image_generation_config(
        model=routed_model,
        provider=LlmProviders.SEEGEN,
    )

    assert routed_model == "seedream-v4.0"
    assert provider == "seegen"
    assert isinstance(config, SeeGenImageGenerationConfig)

    image_prices = {
        "seegen/seedream-v4.0": 0.03,
        "seegen/seedream-v5.0-lite": 0.035,
        "seegen/seedream-v4.5": 0.04,
        "seegen/seedream-v5.0-pro": 0.045,
        "seegen/nano-banana-2": 0.08,
        "seegen/nano-banana-pro": 0.16,
    }
    token_models = (
        "seegen/gpt-image-2",
        "seegen/gpt-image-2.5-sunburst",
        "seegen/gpt-image-2.5-flare",
    )
    repository_root = Path(__file__).resolve().parents[5]
    for relative_path in (
        "model_prices_and_context_window.json",
        "litellm/model_prices_and_context_window_backup.json",
    ):
        prices = TypeAdapter(dict[str, dict[str, PydanticJsonValue]]).validate_json(
            (repository_root / relative_path).read_bytes()
        )
        assert {model: prices[model]["output_cost_per_image"] for model in image_prices} == image_prices
        for model in token_models:
            assert prices[model]["input_cost_per_token"] == 0.000005
            assert prices[model]["cache_read_input_token_cost"] == 0.00000125
            assert prices[model]["input_cost_per_image_token"] == 0.000008
            assert "cache_read_input_image_token_cost" not in prices[model]
            assert prices[model]["output_cost_per_image_token"] == 0.00003


def test_cost_router_uses_flat_and_token_pricing(local_model_cost_map: None) -> None:
    flat_cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="seedream-v4.0",
        custom_llm_provider="seegen",
        completion_response=ImageResponse(data=[ImageObject(url="https://cdn.example.com/seedream.png")]),
    )
    token_cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="gpt-image-2",
        custom_llm_provider="seegen",
        completion_response=ImageResponse(
            data=[ImageObject(url="https://cdn.example.com/gpt.png")],
            usage=ImageUsage(
                input_tokens=20,
                input_tokens_details=ImageUsageInputTokensDetails(text_tokens=20, image_tokens=0),
                output_tokens=100,
                total_tokens=120,
            ),
        ),
    )

    assert math.isclose(flat_cost, 0.03)
    assert math.isclose(token_cost, 0.0031)
