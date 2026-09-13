import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.seegen.common_utils import SeeGenError
from litellm.llms.seegen.videos import (
    SeeGenDashScopeVideoConfig,
    SeeGenSeedanceVideoConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)
from litellm.utils import ProviderConfigManager
from litellm.videos.capabilities import DeclaredCapabilityParams

API_BASE = "https://api.seegen.ai"
SEEDANCE_MODEL = "doubao-seedance-2-5-260628"
HAPPYHORSE_MODEL = "happyhorse-1.1-r2v"
WAN_MODEL = "wan3.0-video"


def _response(payload: dict[str, JsonValue], status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", f"{API_BASE}/task")
    return httpx.Response(status_code, json=payload, request=request)


def _transform_create(config, model: str, prompt: str, params: dict[str, JsonValue]):
    mapped = config.map_openai_params(
        video_create_optional_params=params,
        model=model,
        drop_params=False,
    )
    return config.transform_video_create_request(
        model=model,
        prompt=prompt,
        api_base=API_BASE,
        video_create_optional_request_params=mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )


def test_provider_config_dispatches_each_seegen_video_family() -> None:
    seedance = ProviderConfigManager.get_provider_video_config(
        model="nsfw-seedance-2-5",
        provider=LlmProviders.SEEGEN,
    )
    happyhorse = ProviderConfigManager.get_provider_video_config(
        model="happyhorse-1.0-video-edit",
        provider=LlmProviders.SEEGEN,
    )
    wan = ProviderConfigManager.get_provider_video_config(
        model="nsfw-wan3.0-video-prime",
        provider=LlmProviders.SEEGEN,
    )

    assert isinstance(seedance, SeeGenSeedanceVideoConfig)
    assert isinstance(happyhorse, SeeGenDashScopeVideoConfig)
    assert isinstance(wan, SeeGenDashScopeVideoConfig)


def test_environment_reuses_seegen_auth_and_family_specific_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SEEGEN_API_KEY", raising=False)
    seedance_headers = SeeGenSeedanceVideoConfig().validate_environment(
        headers={}, model=SEEDANCE_MODEL, api_key="secret"
    )
    happyhorse_headers = SeeGenDashScopeVideoConfig(HAPPYHORSE_MODEL).validate_environment(
        headers={}, model=HAPPYHORSE_MODEL, api_key="secret"
    )
    wan_headers = SeeGenDashScopeVideoConfig(WAN_MODEL).validate_environment(
        headers={}, model=WAN_MODEL, api_key="secret"
    )

    assert seedance_headers == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
    }
    assert happyhorse_headers["X-DashScope-Async"] == "enable"
    assert "X-DashScope-Async" not in wan_headers


def test_seedance_request_maps_openai_params_and_reference_roles() -> None:
    config = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL)
    body, files, url = _transform_create(
        config,
        SEEDANCE_MODEL,
        "A horse running through fog",
        {
            "seconds": "12",
            "size": "1280x720",
            "image_url": "https://assets.example/start.png",
            "end_image_url": "https://assets.example/end.png",
            "input_reference": ["https://assets.example/ref-a.png"],
            "image_urls": ["https://assets.example/ref-b.png"],
            "video_urls": ["https://assets.example/ref.mp4"],
            "audio_urls": ["https://assets.example/ref.mp3"],
            "generate_audio": False,
            "bitrate_mode": "high",
            "output_format": "mov",
            "omni_reference_task_type": "reference",
        },
    )

    assert url == f"{API_BASE}/v1/contents/generations/tasks"
    assert files == []
    assert body == {
        "model": SEEDANCE_MODEL,
        "content": [
            {"type": "text", "text": "A horse running through fog"},
            {
                "type": "image_url",
                "image_url": {"url": "https://assets.example/start.png"},
                "role": "first_frame",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://assets.example/end.png"},
                "role": "last_frame",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://assets.example/ref-a.png"},
                "role": "reference_image",
            },
            {
                "type": "image_url",
                "image_url": {"url": "https://assets.example/ref-b.png"},
                "role": "reference_image",
            },
            {
                "type": "video_url",
                "video_url": {"url": "https://assets.example/ref.mp4"},
                "role": "reference_video",
            },
            {
                "type": "audio_url",
                "audio_url": {"url": "https://assets.example/ref.mp3"},
                "role": "reference_audio",
            },
        ],
        "generate_audio": False,
        "ratio": "16:9",
        "duration": 12,
        "resolution": "720p",
        "bitrate_mode": "high",
        "output_format": "mov",
        "omni_reference_task_type": "reference",
    }
    assert "watermark" not in body
    assert "seed" not in body


def test_seedance_rejects_seed_but_passes_upscale_resolutions() -> None:
    config = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL)

    with pytest.raises(SeeGenError, match="seed"):
        config.map_openai_params(
            video_create_optional_params={"seed": 7},
            model=SEEDANCE_MODEL,
            drop_params=False,
        )

    mapped = config.map_openai_params(
        video_create_optional_params={"resolution": "4K"},
        model=SEEDANCE_MODEL,
        drop_params=False,
    )
    assert mapped["resolution"] == "4K"


def test_seedance_edit_forces_auto_duration() -> None:
    config = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL)
    mapped = config.map_openai_params(
        video_create_optional_params={"omni_reference_task_type": "edit"},
        model=SEEDANCE_MODEL,
        drop_params=False,
    )

    assert mapped["duration"] == -1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("queued", "processing"),
        ("running", "processing"),
        ("succeeded", "completed"),
        ("failed", "failed"),
        ("cancelled", "failed"),
        ("expired", "failed"),
    ],
)
def test_seedance_status_mapping_and_token_usage(status: str, expected: str) -> None:
    config = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL)
    status_response = config.transform_video_status_retrieve_response(
        raw_response=_response(
            {
                "id": "cgt-123",
                "status": status,
                "usage": {"completion_tokens": 250, "total_tokens": 300},
            }
        ),
        logging_obj=Mock(),
        custom_llm_provider="seegen",
    )

    assert status_response.status == expected
    assert status_response.usage == {"completion_tokens": 250, "total_tokens": 300}


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "video_url", "video_url": {"url": "https://cdn.example/result.mp4"}}],
        {"video_url": "https://cdn.example/result.mp4"},
    ],
)
def test_seedance_content_download_parses_both_vendor_shapes(content: JsonValue) -> None:
    def route(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://cdn.example/result.mp4"
        return httpx.Response(200, content=b"seedance-video", request=request)

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(route)))
    config = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL, sync_client=client)

    result = config.transform_video_content_response(
        raw_response=_response({"id": "cgt-123", "status": "succeeded", "content": content}),
        logging_obj=Mock(),
    )

    assert result == b"seedance-video"


def test_happyhorse_request_shapes_and_forced_watermark() -> None:
    t2v_body, _, _ = _transform_create(
        SeeGenDashScopeVideoConfig("happyhorse-1.1-t2v"),
        "happyhorse-1.1-t2v",
        "A quiet shoreline",
        {"seconds": "6", "size": "1920x1080", "seed": 42, "watermark": True},
    )
    i2v_body, _, _ = _transform_create(
        SeeGenDashScopeVideoConfig("happyhorse-1.1-i2v"),
        "happyhorse-1.1-i2v",
        "Animate the frame",
        {"image_url": "https://assets.example/start.png"},
    )
    r2v_body, _, _ = _transform_create(
        SeeGenDashScopeVideoConfig(HAPPYHORSE_MODEL),
        HAPPYHORSE_MODEL,
        "Keep the same horse",
        {"input_reference": ["https://assets.example/a.png", "https://assets.example/b.png"]},
    )
    edit_body, _, _ = _transform_create(
        SeeGenDashScopeVideoConfig("happyhorse-1.0-video-edit"),
        "happyhorse-1.0-video-edit",
        "Restyle the clip",
        {
            "video_urls": ["https://assets.example/source.mp4"],
            "input_reference": ["https://assets.example/style.png"],
            "audio_setting": "origin",
            "resolution": "1080P",
        },
    )

    assert t2v_body == {
        "model": "happyhorse-1.1-t2v",
        "input": {"prompt": "A quiet shoreline"},
        "parameters": {
            "resolution": "1080P",
            "ratio": "16:9",
            "duration": 6,
            "seed": 42,
            "watermark": False,
        },
    }
    assert i2v_body["input"]["media"] == [{"type": "first_frame", "url": "https://assets.example/start.png"}]
    assert r2v_body["input"]["media"] == [
        {"type": "reference_image", "url": "https://assets.example/a.png"},
        {"type": "reference_image", "url": "https://assets.example/b.png"},
    ]
    assert edit_body["input"]["media"] == [
        {"type": "video", "url": "https://assets.example/source.mp4"},
        {"type": "reference_image", "url": "https://assets.example/style.png"},
    ]
    assert edit_body["parameters"]["audio_setting"] == "origin"


def test_wan_request_maps_frames_reference_media_audio_and_seed() -> None:
    config = SeeGenDashScopeVideoConfig(WAN_MODEL)
    body, _, url = _transform_create(
        config,
        WAN_MODEL,
        "A documentary portrait",
        {
            "input_reference": ["https://assets.example/ref.png"],
            "video_urls": ["https://assets.example/ref.mp4"],
            "audio_urls": ["https://assets.example/ref.mp3"],
            "generate_audio": False,
            "seconds": "20",
            "size": "720x1280",
            "seed": -1,
            "prompt_extend": False,
        },
    )

    assert url == f"{API_BASE}/api/v1/services/aigc/video-generation/video-synthesis"
    assert body == {
        "model": WAN_MODEL,
        "input": {
            "prompt": "A documentary portrait",
            "media": [
                {"type": "reference_image", "url": "https://assets.example/ref.png"},
                {"type": "reference_video", "url": "https://assets.example/ref.mp4"},
                {"type": "reference_audio", "url": "https://assets.example/ref.mp3"},
            ],
        },
        "parameters": {
            "resolution": "720P",
            "ratio": "9:16",
            "duration": 20,
            "audio": False,
            "seed": -1,
            "prompt_extend": False,
        },
    }


def test_wan_rejects_mixed_frame_and_reference_modes() -> None:
    with pytest.raises(SeeGenError, match="mutually exclusive"):
        _transform_create(
            SeeGenDashScopeVideoConfig(WAN_MODEL),
            WAN_MODEL,
            "prompt",
            {
                "image_url": "https://assets.example/start.png",
                "video_urls": ["https://assets.example/ref.mp4"],
            },
        )


def test_wan_frame_mode_maps_first_and_last_frame() -> None:
    body, _, _ = _transform_create(
        SeeGenDashScopeVideoConfig(WAN_MODEL),
        WAN_MODEL,
        "Bridge the two keyframes",
        {
            "image_url": "https://assets.example/start.png",
            "end_image_url": "https://assets.example/end.png",
        },
    )

    assert body["input"]["media"] == [
        {"type": "first_frame", "url": "https://assets.example/start.png"},
        {"type": "last_frame", "url": "https://assets.example/end.png"},
    ]


def test_wan_promptless_requests_require_media() -> None:
    config = SeeGenDashScopeVideoConfig(WAN_MODEL)

    with pytest.raises(SeeGenError, match="prompt or media"):
        _transform_create(config, WAN_MODEL, "", {})

    body, _, _ = _transform_create(
        config,
        WAN_MODEL,
        "",
        {"image_url": "https://assets.example/start.png"},
    )
    assert body["input"]["media"] == [{"type": "first_frame", "url": "https://assets.example/start.png"}]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("PENDING", "processing"),
        ("RUNNING", "processing"),
        ("SUCCEEDED", "completed"),
        ("FAILED", "failed"),
        ("CANCELED", "failed"),
        ("UNKNOWN", "failed"),
    ],
)
def test_dashscope_status_mapping_uses_output_message_and_actual_usage(status: str, expected: str) -> None:
    config = SeeGenDashScopeVideoConfig(WAN_MODEL)
    video = config.transform_video_status_retrieve_response(
        raw_response=_response(
            {
                "request_id": "req-1",
                "output": {
                    "task_id": "task-1",
                    "task_status": status,
                    "message": "render failed" if status == "FAILED" else None,
                    "video_url": "https://cdn.example/video.mp4" if status == "SUCCEEDED" else None,
                },
                "usage": {"output_video_duration": 8, "SR": "1080P"},
            }
        ),
        logging_obj=Mock(),
        custom_llm_provider="seegen",
    )

    assert video.status == expected
    assert video.usage == {"duration_seconds": 8.0, "video_resolution": "1080p"}
    if status == "FAILED":
        assert video.error == {"code": "failed", "message": "render failed"}
    if status == "UNKNOWN":
        assert video.error is not None
        assert "expired or was not found" in video.error["message"]


@pytest.mark.parametrize(
    ("model", "supported"),
    [
        (
            SEEDANCE_MODEL,
            {
                "image_url",
                "end_image_url",
                "input_reference",
                "image_urls",
                "video_urls",
                "audio_urls",
                "bitrate_mode",
                "generate_audio",
            },
        ),
        ("happyhorse-1.1-t2v", set()),
        ("happyhorse-1.1-i2v", {"image_url"}),
        ("happyhorse-1.1-r2v", {"input_reference"}),
        ("happyhorse-1.0-video-edit", {"input_reference", "video_urls", "base_video_url"}),
        (
            WAN_MODEL,
            {
                "image_url",
                "end_image_url",
                "input_reference",
                "video_urls",
                "audio_urls",
                "generate_audio",
            },
        ),
    ],
)
def test_capability_declarations_are_model_specific(model: str, supported: set[str]) -> None:
    config = SeeGenSeedanceVideoConfig(model) if "seedance" in model else SeeGenDashScopeVideoConfig(model)
    declaration = config.get_capability_param_support(model)

    assert isinstance(declaration, DeclaredCapabilityParams)
    assert declaration.supported == frozenset(supported)


def test_submit_responses_encode_provider_and_model() -> None:
    seedance = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL).transform_video_create_response(
        model=SEEDANCE_MODEL,
        raw_response=_response({"id": "cgt-123"}),
        logging_obj=Mock(),
        custom_llm_provider="seegen",
    )
    dashscope = SeeGenDashScopeVideoConfig(HAPPYHORSE_MODEL).transform_video_create_response(
        model=HAPPYHORSE_MODEL,
        raw_response=_response({"request_id": "req-1", "output": {"task_id": "task-1", "task_status": "PENDING"}}),
        logging_obj=Mock(),
        custom_llm_provider="seegen",
    )

    assert decode_video_id_with_provider(seedance.id) == {
        "custom_llm_provider": "seegen",
        "model_id": SEEDANCE_MODEL,
        "video_id": "cgt-123",
    }
    assert decode_video_id_with_provider(dashscope.id) == {
        "custom_llm_provider": "seegen",
        "model_id": HAPPYHORSE_MODEL,
        "video_id": "task-1",
    }


@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [(402, "insufficient_balance"), (429, "rate_limit_exceeded")],
)
def test_video_submit_preserves_account_error_status(status_code: int, error_code: str) -> None:
    response = _response(
        {"error": error_code, "message": "vendor account error"},
        status_code=status_code,
    )

    with pytest.raises(SeeGenError) as exc_info:
        SeeGenSeedanceVideoConfig(SEEDANCE_MODEL).transform_video_create_response(
            model=SEEDANCE_MODEL,
            raw_response=response,
            logging_obj=Mock(),
        )

    assert exc_info.value.status_code == status_code
    assert error_code in exc_info.value.message


def test_seedance_cancel_preserves_retryable_conflict_status() -> None:
    response = _response(
        {"error": "task_conflict", "message": "task is changing state"},
        status_code=409,
    )

    with pytest.raises(SeeGenError) as exc_info:
        SeeGenSeedanceVideoConfig(SEEDANCE_MODEL).transform_video_delete_response(
            raw_response=response,
            logging_obj=Mock(),
        )

    assert exc_info.value.status_code == 409


def test_seedance_cancel_returns_the_cancelled_task_id() -> None:
    config = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL)
    video_id = encode_video_id_with_provider("cgt-cancel", "seegen", SEEDANCE_MODEL)
    url, data = config.transform_video_delete_request(
        video_id=video_id,
        api_base=API_BASE,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    response = httpx.Response(204, request=httpx.Request("DELETE", url))

    cancelled = config.transform_video_delete_response(response, Mock())

    assert url == f"{API_BASE}/v1/contents/generations/tasks/cgt-cancel"
    assert data == {}
    assert cancelled.id == "cgt-cancel"
    assert cancelled.status == "cancelled"


def test_dashscope_malformed_submit_is_reported_as_upstream_malformed() -> None:
    with pytest.raises(SeeGenError, match="upstream_malformed") as exc_info:
        SeeGenDashScopeVideoConfig(HAPPYHORSE_MODEL).transform_video_create_response(
            model=HAPPYHORSE_MODEL,
            raw_response=_response({"request_id": "req-without-output"}),
            logging_obj=Mock(),
        )

    assert exc_info.value.status_code == 502


def test_task_urls_decode_litellm_video_ids() -> None:
    seedance_id = encode_video_id_with_provider("cgt-123", "seegen", SEEDANCE_MODEL)
    dashscope_id = encode_video_id_with_provider("task-1", "seegen", HAPPYHORSE_MODEL)

    seedance_url, _ = SeeGenSeedanceVideoConfig(SEEDANCE_MODEL).transform_video_status_retrieve_request(
        video_id=seedance_id,
        api_base=API_BASE,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    dashscope_url, _ = SeeGenDashScopeVideoConfig(HAPPYHORSE_MODEL).transform_video_content_request(
        video_id=dashscope_id,
        api_base=API_BASE,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert seedance_url == f"{API_BASE}/v1/contents/generations/tasks/cgt-123"
    assert dashscope_url == f"{API_BASE}/api/v1/tasks/task-1"


def test_all_video_prices_are_registered_in_both_cost_maps() -> None:
    seedance_prices = {
        "doubao-seedance-2-5-260628": 6e-06,
        "doubao-seedance-2-0-260128": 4e-06,
        "doubao-seedance-2-0-fast-260128": 3.142857e-06,
        "doubao-seedance-2-0-mini-260615": 2e-06,
        "dreamina-seedance-2-5-260628": 6.4e-06,
        "dreamina-seedance-2-0-260128": 4.3e-06,
        "dreamina-seedance-2-0-fast-260128": 3.3e-06,
        "dreamina-seedance-2-0-mini-260615": 2.1e-06,
        "nsfw-seedance-2-5": 6.4e-06,
        "nsfw-seedance-2-0": 4.3e-06,
        "nsfw-seedance-2-0-fast": 3.3e-06,
        "nsfw-seedance-2-0-mini": 2.1e-06,
    }
    per_second_prices = {
        "happyhorse-1.1-t2v": (0.1285714, 0.1714286),
        "happyhorse-1.1-i2v": (0.1285714, 0.1714286),
        "happyhorse-1.1-r2v": (0.1285714, 0.1714286),
        "happyhorse-1.0-t2v": (0.1285714, 0.2285714),
        "happyhorse-1.0-i2v": (0.1285714, 0.2285714),
        "happyhorse-1.0-r2v": (0.1285714, 0.2285714),
        "happyhorse-1.0-video-edit": (0.1285714, 0.2285714),
    }
    wan_prices = {
        "wan3.0-video": 0.0428571,
        "nsfw-wan3.0-video": 0.0428571,
        "wan3.0-video-prime": 0.0642857,
        "nsfw-wan3.0-video-prime": 0.0642857,
    }
    repository_root = Path(__file__).resolve().parents[5]

    for relative_path in (
        "model_prices_and_context_window.json",
        "litellm/model_prices_and_context_window_backup.json",
    ):
        prices = TypeAdapter(dict[str, dict[str, JsonValue]]).validate_json(
            (repository_root / relative_path).read_bytes()
        )
        for model, rate in seedance_prices.items():
            entry = prices[f"seegen/{model}"]
            assert entry["mode"] == "video_generation"
            assert entry["output_cost_per_video_token"] == rate
        for model, (base_rate, high_rate) in per_second_prices.items():
            entry = prices[f"seegen/{model}"]
            assert entry["output_cost_per_second"] == base_rate
            assert entry["output_cost_per_second_1080p"] == high_rate
            if model.startswith("happyhorse-1.1"):
                assert entry["output_cost_per_second_480p"] == 0.0642857
        for model, rate in wan_prices.items():
            assert prices[f"seegen/{model}"]["output_cost_per_second"] == rate


def test_public_video_generation_posts_seedance_json() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{API_BASE}/v1/contents/generations/tasks"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert json.loads(request.content) == {
            "model": SEEDANCE_MODEL,
            "content": [{"type": "text", "text": "A lighthouse in a storm"}],
            "duration": 8,
        }
        return httpx.Response(200, json={"id": "cgt-public"}, request=request)

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(route)))

    video = litellm.video_generation(
        prompt="A lighthouse in a storm",
        model=f"seegen/{SEEDANCE_MODEL}",
        seconds="8",
        api_key="test-key",
        client=client,
        timeout=1,
    )

    assert video.status == "queued"
    assert decode_video_id_with_provider(video.id)["video_id"] == "cgt-public"


def test_public_video_generation_posts_happyhorse_json_and_async_header() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == f"{API_BASE}/api/v1/services/aigc/video-generation/video-synthesis"
        assert request.headers["X-DashScope-Async"] == "enable"
        assert json.loads(request.content) == {
            "model": "happyhorse-1.1-t2v",
            "input": {"prompt": "A paper boat on a stream"},
            "parameters": {"duration": 5, "seed": 9, "watermark": False},
        }
        return httpx.Response(
            200,
            json={"request_id": "req-public", "output": {"task_id": "task-public", "task_status": "PENDING"}},
            request=request,
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(route)))

    video = litellm.video_generation(
        prompt="A paper boat on a stream",
        model="seegen/happyhorse-1.1-t2v",
        seconds="5",
        seed=9,
        api_key="test-key",
        client=client,
        timeout=1,
    )

    assert video.status == "processing"
    assert decode_video_id_with_provider(video.id)["video_id"] == "task-public"


def test_public_video_content_polls_wan_then_downloads_result_bytes() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.seegen.ai":
            assert str(request.url) == f"{API_BASE}/api/v1/tasks/task-content"
            assert "X-DashScope-Async" not in request.headers
            return httpx.Response(
                200,
                json={
                    "output": {
                        "task_id": "task-content",
                        "task_status": "SUCCEEDED",
                        "video_url": "https://cdn.example/result.mp4",
                    }
                },
                request=request,
            )
        assert str(request.url) == "https://cdn.example/result.mp4"
        return httpx.Response(200, content=b"wan-video", request=request)

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(route)))
    video_id = encode_video_id_with_provider("task-content", "seegen", WAN_MODEL)

    content = litellm.video_content(
        video_id=video_id,
        api_key="test-key",
        client=client,
        timeout=1,
    )

    assert content == b"wan-video"


def test_dashscope_status_parses_the_real_integer_sr_usage() -> None:
    config = SeeGenDashScopeVideoConfig(WAN_MODEL)
    video = config.transform_video_status_retrieve_response(
        raw_response=_response(
            {
                "request_id": "f8f86eb3-7080-4e6f-a08a-f7ba60fe2c23",
                "output": {
                    "task_id": "92a46dbb-e299-4579-89b2-dcc806591b32",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://dashscope-0816.oss-accelerate.aliyuncs.com/result.mp4",
                    "original_video_url": "https://dashscope-0816.oss-accelerate.aliyuncs.com/result.mp4",
                    "orig_prompt": "a paper boat drifting through a puddle",
                    "submit_time": "2026-09-13 17:49:04.480",
                    "scheduled_time": "2026-09-13 17:49:04.498",
                    "end_time": "2026-09-13 17:50:47.689",
                },
                "usage": {
                    "SR": 480,
                    "fps": 30,
                    "ratio": "16:9",
                    "duration": 3,
                    "video_count": 1,
                    "input_video_duration": 0,
                    "output_video_duration": 3,
                },
            }
        ),
        logging_obj=Mock(),
        custom_llm_provider="seegen",
    )

    assert video.status == "completed"
    assert video.usage == {"duration_seconds": 3.0, "video_resolution": "480p"}
