from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.fal_ai.videos.transformation import FalAIVideoConfig, _classify_result_payload, _GeneratedVideo
from litellm.types.router import GenericLiteLLMParams
from litellm.videos.capabilities import DeclaredCapabilityParams

MODEL = "bria/video/background-removal/v3"


def request(params, model=MODEL):
    config = FalAIVideoConfig()
    mapped = config.map_openai_params(params, model, False)
    return config.transform_video_create_request(
        model, "unused", "https://queue.fal.run", mapped, GenericLiteLLMParams(), {}
    )


@pytest.mark.parametrize("model", [MODEL, f"fal_ai/{MODEL}"])
def test_promptless_reference_capabilities_and_alpha_defaults(model):
    config = FalAIVideoConfig()
    assert config.supports_promptless_video_create(model)
    assert config.get_capability_param_support(model) == DeclaredCapabilityParams(frozenset(("input_reference",)))
    body, files, url = request({"input_reference": "https://example.com/source.mp4", "seconds": "5"}, model)
    assert url == f"https://queue.fal.run/{MODEL}"
    assert body["video_url"] == "https://example.com/source.mp4"
    assert body["background_color"] == "Transparent"
    assert body["output_container_and_codec"] == "webm_vp9"
    assert "prompt" not in body
    assert files == []


@pytest.mark.parametrize("override", [{"background_color": "Black"}, {"output_container_and_codec": "mov_proresks"}])
def test_explicit_alpha_overrides(override):
    body, _, _ = request({"seconds": "5", "extra_body": {"video_url": "https://example.com/source.mp4", **override}})
    for key, value in override.items():
        assert body[key] == value


@pytest.mark.parametrize(
    "params",
    [
        {"input_reference": "data:video/mp4;base64,AAAA"},
        {"extra_body": {"video_url": "data:video/mp4;base64,AAAA"}},
        {"input_reference": b"clip"},
    ],
)
def test_data_uri_rejected(params):
    with pytest.raises(ValueError, match="hosted.*video_url"):
        request({"seconds": "5", **params})


@pytest.mark.parametrize("seconds", [None, "0", "-1", "nan", "inf", "invalid"])
def test_missing_or_invalid_seconds_rejected(seconds):
    with pytest.raises(ValueError, match="seconds"):
        request({"input_reference": "https://example.com/source.mp4", "seconds": seconds})


def test_cost_uses_source_seconds(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    body, _, _ = request({"input_reference": "https://example.com/source.mp4", "seconds": "5"})
    response = FalAIVideoConfig().transform_video_create_response(
        MODEL,
        httpx.Response(200, json={"request_id": "test", "status": "IN_QUEUE"}),
        Mock(),
        "fal_ai",
        body,
    )
    assert response.usage["duration_seconds"] == 5
    assert litellm.completion_cost(
        completion_response=response, model=MODEL, custom_llm_provider="fal_ai", call_type="create_video"
    ) == pytest.approx(0.25)


def test_request_without_duration_is_refused_before_submission():
    """The NOL-519 $0-COGS class is closed at the request boundary, not at costing time.

    fal bills this route per second of SOURCE clip and its queue response carries no
    duration, so a submission that omits `seconds` could only ever record $0. Refusing
    it here means the job is never created; catching it in the cost calculator would
    fire only after the provider had already rendered and billed us.
    """
    with pytest.raises(ValueError, match="seconds"):
        request({"input_reference": "https://example.com/source.mp4"})


def test_null_content_type_and_vendor_download_url():
    url = "https://temp.bria.ai/result.webm"
    assert _classify_result_payload(
        {"video": {"url": url, "content_type": None}, "warning": "notice"}
    ) == _GeneratedVideo(url)


@pytest.mark.parametrize(
    "param,value",
    [
        ("size", "1280x720"),
        ("size", "auto"),
        ("aspect_ratio", "16:9"),
        ("resolution", "1080p"),
        ("target_resolution", "4k"),
    ],
)
def test_unsupported_dimensions_rejected(param, value):
    with pytest.raises(ValueError, match="does not support"):
        request({"seconds": "5", "input_reference": "https://example.com/source.mp4", param: value})


@pytest.mark.parametrize("param", ["image_url", "end_image_url", "generate_audio", "audio_urls"])
def test_unsupported_capability_is_refused(param):
    from litellm.videos.capabilities import check_capability_params

    failure = check_capability_params(
        MODEL, "fal_ai", FalAIVideoConfig().get_capability_param_support(MODEL), {param: "requested"}
    )
    assert failure is not None
    assert failure.requested == (param,)
