from unittest.mock import AsyncMock, Mock

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
def test_promptless_reference_capabilities(model):
    config = FalAIVideoConfig()
    assert config.supports_promptless_video_create(model)
    assert config.get_capability_param_support(model) == DeclaredCapabilityParams(frozenset(("input_reference",)))


@pytest.mark.parametrize("model", [MODEL, f"fal_ai/{MODEL}"])
@pytest.mark.parametrize("seconds", [None, "0", "-1", "nan", "inf", "invalid", "1", "60"])
def test_unverified_source_duration_is_refused_before_submission(model, seconds):
    with pytest.raises(litellm.BadRequestError, match="source duration is verified"):
        request({"input_reference": "https://example.com/source.mp4", "seconds": seconds}, model)


@pytest.mark.parametrize(
    "extra_body",
    [
        {"video_url": "https://example.com/source.mp4", "duration": "1"},
        {"video_url": "https://example.com/source.mp4", "duration": "60", "trusted_seconds": 60},
        {"video_url": "https://example.com/source.mp4", "output_container_and_codec": "mov_proresks"},
    ],
)
def test_provider_overrides_cannot_bypass_source_duration_gate(extra_body):
    with pytest.raises(litellm.BadRequestError, match="source duration is verified"):
        request({"seconds": "60", "extra_body": extra_body})


@pytest.mark.parametrize("use_async", (False, True))
@pytest.mark.asyncio
async def test_public_creation_rejects_untrusted_duration_without_posting(monkeypatch, use_async):
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    sync_post = Mock()
    async_post = AsyncMock()
    monkeypatch.setattr(HTTPHandler, "post", sync_post)
    monkeypatch.setattr(AsyncHTTPHandler, "post", async_post)
    with pytest.raises(litellm.BadRequestError, match="source duration is verified") as error:
        if use_async:
            await litellm.avideo_generation(
                model=f"fal_ai/{MODEL}", api_key="test-key",
                input_reference="https://example.com/sixty-seconds.mp4", seconds="1",
            )
        else:
            litellm.video_generation(
                model=f"fal_ai/{MODEL}", api_key="test-key",
                input_reference="https://example.com/sixty-seconds.mp4", seconds="1",
            )
    assert error.value.status_code == 400
    sync_post.assert_not_called()
    async_post.assert_not_called()


def test_existing_background_removal_job_status_still_resolves():
    from litellm.types.videos.utils import encode_video_id_with_provider

    video_id = encode_video_id_with_provider("existing-job", "fal_ai", MODEL)
    url, params = FalAIVideoConfig().transform_video_status_retrieve_request(
        video_id, "https://queue.fal.run", GenericLiteLLMParams(), {}
    )
    assert url == "https://queue.fal.run/bria/video/requests/existing-job/status"
    assert params == {}


def test_null_content_type_and_vendor_download_url():
    url = "https://temp.bria.ai/result.webm"
    assert _classify_result_payload(
        {"video": {"url": url, "content_type": None}, "warning": "notice"}
    ) == _GeneratedVideo(url)


@pytest.mark.parametrize("param", ["image_url", "end_image_url", "generate_audio", "audio_urls"])
def test_unsupported_capability_is_refused(param):
    from litellm.videos.capabilities import check_capability_params

    failure = check_capability_params(
        MODEL, "fal_ai", FalAIVideoConfig().get_capability_param_support(MODEL), {param: "requested"}
    )
    assert failure is not None
    assert failure.requested == (param,)
