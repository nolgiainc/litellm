import pytest

from litellm.llms.seegen.common_utils import JsonValue, SeeGenError
from litellm.llms.seegen.image_generation.transformation import SeeGenImageGenerationConfig


@pytest.fixture
def config() -> SeeGenImageGenerationConfig:
    return SeeGenImageGenerationConfig()


def test_seedream_maps_openai_fields_and_forces_platform_defaults(config: SeeGenImageGenerationConfig) -> None:
    mapped = config.map_openai_params(
        non_default_params={
            "size": "4K",
            "image": ["https://example.com/one.png", "data:image/png;base64,dHdv"],
            "watermark": True,
            "response_format": "b64_json",
        },
        optional_params={},
        model="seedream-v5.0-lite",
        drop_params=False,
    )

    result = config.transform_image_generation_request(
        model="seedream-v5.0-lite",
        prompt="draw a lighthouse",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert result == {
        "model": "seedream-v5.0-lite",
        "prompt": "draw a lighthouse",
        "image": ["https://example.com/one.png", "data:image/png;base64,dHdv"],
        "size": "4K",
        "response_format": "url",
        "watermark": False,
    }


def test_seedream_defaults_size_and_drops_unsupported_fields(config: SeeGenImageGenerationConfig) -> None:
    mapped = config.map_openai_params(
        non_default_params={"n": 2, "output_format": "png"},
        optional_params={},
        model="seedream-v4.5",
        drop_params=True,
    )

    result = config.transform_image_generation_request(
        model="seedream-v4.5",
        prompt="draw a lighthouse",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert result["size"] == "2048x2048"
    assert "n" not in result
    assert "output_format" not in result


@pytest.mark.parametrize(
    ("model", "params", "message"),
    [
        ("seedream-v5.0-pro", {"sequential_image_generation": "auto"}, "sequential_image_generation"),
        ("seedream-v4.0", {"response_format": "json"}, "response_format"),
        ("nano-banana-2", {"response_format": "json"}, "response_format"),
    ],
)
def test_explicit_families_reject_unsupported_modes(
    config: SeeGenImageGenerationConfig,
    model: str,
    params: dict[str, JsonValue],
    message: str,
) -> None:
    with pytest.raises(SeeGenError, match=message):
        config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model=model,
            drop_params=False,
        )


def test_gpt_image_headers_have_one_stable_idempotency_key(config: SeeGenImageGenerationConfig) -> None:
    headers = config.validate_environment(
        headers={},
        model="gpt-image-2.5-flare",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    transformed = config.transform_image_generation_request(
        model="gpt-image-2.5-flare",
        prompt="draw a lighthouse",
        optional_params={"quality": "xhigh", "n": 2, "stream": False},
        litellm_params={},
        headers=headers,
    )

    assert headers["Idempotency-Key"]
    assert transformed["quality"] == "xhigh"
    assert "stream" not in transformed
    assert (
        config.transform_image_generation_request(
            model="gpt-image-2.5-flare",
            prompt="draw a lighthouse",
            optional_params={"quality": "xhigh", "n": 2},
            litellm_params={},
            headers=headers,
        )
        == transformed
    )


def test_gpt_image_preserves_caller_idempotency_key_case_insensitively(config: SeeGenImageGenerationConfig) -> None:
    headers = config.validate_environment(
        headers={"idempotency-key": "caller-key"},
        model="gpt-image-2",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    idempotency_headers = {key: value for key, value in headers.items() if key.lower() == "idempotency-key"}
    assert idempotency_headers == {"Idempotency-Key": "caller-key"}


def test_environment_key_is_resolved_through_secret_manager(
    config: SeeGenImageGenerationConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEEGEN_API_KEY", "environment-key")

    headers = config.validate_environment(
        headers={},
        model="nano-banana-2",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer environment-key"


@pytest.mark.parametrize("param,value", [("stream", True), ("partial_images", 0), ("partial_images", 1)])
def test_gpt_image_rejects_async_incompatible_fields(
    config: SeeGenImageGenerationConfig,
    param: str,
    value: bool | int,
) -> None:
    with pytest.raises(SeeGenError, match=param):
        config.map_openai_params(
            non_default_params={param: value},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


def test_gpt_image_strips_async_incompatible_fields_with_drop_params(config: SeeGenImageGenerationConfig) -> None:
    result = config.map_openai_params(
        non_default_params={"stream": True, "partial_images": 2, "quality": "high"},
        optional_params={},
        model="gpt-image-2",
        drop_params=True,
    )

    assert result == {"quality": "high"}


def test_gpt_image_xhigh_is_limited_to_2_5_models(config: SeeGenImageGenerationConfig) -> None:
    with pytest.raises(SeeGenError, match="quality"):
        config.map_openai_params(
            non_default_params={"quality": "xhigh"},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


def test_gpt_image_passes_unmapped_vendor_fields_verbatim(config: SeeGenImageGenerationConfig) -> None:
    result = config.map_openai_params(
        non_default_params={"vendor_option": "passthrough"},
        optional_params={},
        model="gpt-image-2.5-flare",
        drop_params=False,
    )

    assert result == {"vendor_option": "passthrough"}


@pytest.mark.parametrize(
    "params",
    [
        {"n": 11},
        {"size": "1025x1024"},
        {"size": "4096x1024"},
        {"input_fidelity": "high"},
        {"response_format": "json"},
    ],
)
def test_gpt_image_rejects_invalid_contract_values(
    config: SeeGenImageGenerationConfig,
    params: dict[str, JsonValue],
) -> None:
    with pytest.raises(SeeGenError):
        config.map_openai_params(
            non_default_params=params,
            optional_params={},
            model="gpt-image-2.5-flare",
            drop_params=False,
        )


def test_seedream_drops_unmapped_payload_fields_when_drop_params_is_enabled(
    config: SeeGenImageGenerationConfig,
) -> None:
    result = config.transform_image_generation_request(
        model="seedream-v4.0",
        prompt="draw a lighthouse",
        optional_params={"vendor_unknown": "value"},
        litellm_params={"drop_params": True},
        headers={},
    )

    assert "vendor_unknown" not in result


@pytest.mark.parametrize(
    ("size", "resolution", "aspect_ratio"),
    [("1024x1024", "1K", "1:1"), ("2048x1536", "2K", "4:3"), ("4K", "4K", None)],
)
def test_nano_banana_maps_size_to_resolution_and_aspect_ratio(
    config: SeeGenImageGenerationConfig,
    size: str,
    resolution: str,
    aspect_ratio: str | None,
) -> None:
    mapped = config.map_openai_params(
        non_default_params={"size": size, "image": "https://example.com/reference.png"},
        optional_params={},
        model="nano-banana-2",
        drop_params=False,
    )

    result = config.transform_image_generation_request(
        model="nano-banana-2",
        prompt="draw a lighthouse",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert result["resolution"] == resolution
    assert result["images"] == ["https://example.com/reference.png"]
    assert result.get("aspect_ratio") == aspect_ratio
