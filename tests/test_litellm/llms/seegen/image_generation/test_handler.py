import base64
import json
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.litellm_core_utils.llm_cost_calc.utils import CostCalculatorUtils
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.seegen.common_utils import JsonValue, SeeGenError
from litellm.llms.seegen.image_generation.handler import SeeGenImageGeneration
from litellm.llms.seegen.image_generation.polling import SeeGenPoller, SeeGenPollRequest
from litellm.types.utils import ImageResponse


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> HTTPHandler:
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _generate(client: HTTPHandler, *, response_format: str = "url") -> ImageResponse:
    result = SeeGenImageGeneration(poll_interval=0, max_polling_time=1).image_generation(
        model="gpt-image-2.5-sunburst",
        prompt="draw a lighthouse",
        model_response=ImageResponse(),
        optional_params={"response_format": response_format},
        litellm_params={"api_key": "test-key", "api_base": "https://api.seegen.ai"},
        logging_obj=MagicMock(),
        timeout=1,
        client=client,
    )
    assert isinstance(result, ImageResponse)
    return result


def test_submit_then_poll_done_returns_urls_and_usage() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/images/generations":
            return httpx.Response(
                202,
                json={
                    "task_id": "img-123",
                    "status": "processing",
                    "model": "gpt-image-2.5-sunburst",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        if request.method == "GET" and request.url.path == "/v1/images/generations/img-123":
            return httpx.Response(
                200,
                json={
                    "task_id": "img-123",
                    "status": "done",
                    "image_urls": ["https://cdn.example.com/one.png", "https://cdn.example.com/two.png"],
                    "usage": {"generated_images": 2, "output_tokens": 200, "total_tokens": 240},
                },
                request=request,
            )
        raise AssertionError((request.method, request.url.path))

    result = _generate(_client(route))

    assert result.data is not None
    assert [image.url for image in result.data] == [
        "https://cdn.example.com/one.png",
        "https://cdn.example.com/two.png",
    ]
    assert result.usage is not None
    assert result.usage.output_tokens == 200
    assert result.usage.total_tokens == 240


def test_b64_response_downloads_completed_images() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/images/generations":
            return httpx.Response(
                202,
                json={
                    "task_id": "img-b64",
                    "status": "processing",
                    "model": "gpt-image-2.5-sunburst",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        if request.method == "GET" and request.url.path == "/v1/images/generations/img-b64":
            return httpx.Response(
                200,
                json={
                    "task_id": "img-b64",
                    "status": "done",
                    "image_urls": ["https://cdn.example.com/generated.png"],
                    "usage": {"generated_images": 1, "output_tokens": 100, "total_tokens": 120},
                },
                request=request,
            )
        if request.method == "GET" and request.url.path == "/generated.png":
            return httpx.Response(200, content=b"image-bytes", request=request)
        raise AssertionError((request.method, request.url.path))

    result = _generate(_client(route), response_format="b64_json")

    assert result.data is not None
    assert result.data[0].url is None
    assert result.data[0].b64_json == base64.b64encode(b"image-bytes").decode()


def test_poll_failed_surfaces_failure_reason() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "task_id": "img-failed",
                    "status": "processing",
                    "model": "gpt-image-2.5-sunburst",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={"task_id": "img-failed", "status": "failed", "failure_reason": "provider rejected prompt"},
            request=request,
        )

    with pytest.raises(SeeGenError, match="provider rejected prompt"):
        _generate(_client(route))


def test_should_retry_false_is_terminal_without_resubmission() -> None:
    requests = MagicMock()

    def route(request: httpx.Request) -> httpx.Response:
        requests(request.method)
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "task_id": "img-terminal",
                    "status": "processing",
                    "model": "gpt-image-2.5-sunburst",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        return httpx.Response(
            200,
            headers={"x-should-retry": "false"},
            json={"task_id": "img-terminal", "status": "processing", "failure_reason": "route unavailable"},
            request=request,
        )

    with pytest.raises(SeeGenError, match="route unavailable") as exc_info:
        _generate(_client(route))

    assert exc_info.value.status_code == 400
    assert [call.args[0] for call in requests.call_args_list] == ["POST", "GET"]


@pytest.mark.parametrize("status_code", (401, 402, 429, 503))
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.asyncio
async def test_terminal_poll_error_preserves_http_status(status_code: int, asynchronous: bool) -> None:
    response = httpx.Response(
        status_code,
        headers={"x-should-retry": "false"},
        json={"error": "poll_failed", "message": "upstream rejected polling"},
        request=httpx.Request("GET", "https://api.seegen.ai/v1/images/generations/img-terminal"),
    )
    request = SeeGenPollRequest(url=str(response.request.url), headers={}, timeout=1)
    poller = SeeGenPoller(interval=0, max_wait=1)
    client = MagicMock()
    client.get = AsyncMock(return_value=response) if asynchronous else MagicMock(return_value=response)

    with pytest.raises(SeeGenError, match="upstream rejected polling") as exc_info:
        if asynchronous:
            await poller.poll_async(request, client)
        else:
            poller.poll_sync(request, client)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.headers["x-should-retry"] == "false"
    assert exc_info.value.response is response
    client.get.assert_called_once()


def test_gpt_submit_timeout_reuses_the_idempotency_key() -> None:
    submit_outcomes = iter(("timeout", "success"))
    submitted_headers = MagicMock()

    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            submitted_headers(request.headers["Idempotency-Key"])
            if next(submit_outcomes) == "timeout":
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(
                202,
                json={
                    "task_id": "img-retried",
                    "status": "processing",
                    "model": "gpt-image-2.5-sunburst",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "task_id": "img-retried",
                "status": "done",
                "image_urls": ["https://cdn.example.com/generated.png"],
                "usage": {"generated_images": 1, "output_tokens": 100, "total_tokens": 120},
            },
            request=request,
        )

    _generate(_client(route))

    keys = [call.args[0] for call in submitted_headers.call_args_list]
    assert len(keys) == 2
    assert keys[0] == keys[1]


@pytest.mark.parametrize(
    "payload",
    [
        {"error": "image_submit_failed", "message": "upstream rejected submission"},
        {"error": {"code": "invalid_size", "message": "bad size", "param": "size", "type": "invalid_request_error"}},
    ],
)
def test_submit_error_shapes_raise_seegen_error(payload: dict[str, JsonValue]) -> None:
    def route(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, content=json.dumps(payload).encode(), request=request)

    with pytest.raises(SeeGenError) as exc_info:
        _generate(_client(route))

    assert exc_info.value.status_code == 502
    assert "image_submit_failed" in exc_info.value.message or "invalid_size" in exc_info.value.message


@pytest.mark.parametrize("cached_tokens", (0, 5, 13))
def test_poll_done_parses_the_real_gpt_image_usage_shape(cached_tokens: int, local_model_cost_map: None) -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/images/generations":
            return httpx.Response(
                202,
                json={
                    "task_id": "img-ie435a7aa4d50c188d297bb7",
                    "status": "processing",
                    "model": "gpt-image-2",
                    "created_at": "2026-09-13T09:27:45.657Z",
                },
                request=request,
            )
        if request.method == "GET" and request.url.path == "/v1/images/generations/img-ie435a7aa4d50c188d297bb7":
            return httpx.Response(
                200,
                json={
                    "task_id": "img-ie435a7aa4d50c188d297bb7",
                    "status": "done",
                    "model": "gpt-image-2",
                    "image_urls": ["https://image3.example.com/2026/09/13/sailboat_0.png"],
                    "usage": {
                        "model": None,
                        "taskId": "task_cIs5srsk6lgHx8euJeJITfYCDJ8u0jnw",
                        "rawUsage": {
                            "images": 1,
                            "image_count": 1,
                            "input_tokens": 13,
                            "total_tokens": 209,
                            "cached_tokens": cached_tokens,
                            "output_tokens": 196,
                            "input_tokens_details": {"text_tokens": 13, "image_tokens": 0},
                            "output_tokens_details": {"text_tokens": 0, "image_tokens": 196},
                        },
                        "imageCount": 1,
                    },
                    "meta": {"model": "gpt-image-2", "is_fallback": False},
                    "failure_reason": None,
                    "created_at": "2026-09-13T09:27:45.657Z",
                    "completed_at": "2026-09-13T09:28:01.389Z",
                },
                request=request,
            )
        raise AssertionError((request.method, request.url.path))

    result = _generate(_client(route))
    assert [image.url for image in result.data or []] == ["https://image3.example.com/2026/09/13/sailboat_0.png"]
    assert result.usage is not None
    assert result.usage.input_tokens == 13
    assert result.usage.output_tokens == 196
    assert result.usage.total_tokens == 209
    assert result.usage.input_tokens_details.text_tokens == 13
    assert result.usage.input_tokens_details.image_tokens == 0
    assert result.usage.input_tokens_details.cached_tokens == cached_tokens
    cost = CostCalculatorUtils.route_image_generation_cost_calculator(
        model="gpt-image-2.5-sunburst",
        custom_llm_provider="seegen",
        completion_response=result,
    )
    assert cost == pytest.approx((13 - cached_tokens) * 0.000005 + cached_tokens * 0.00000125 + 196 * 0.00003)
