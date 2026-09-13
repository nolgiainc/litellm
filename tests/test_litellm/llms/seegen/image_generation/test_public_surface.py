import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler


def test_image_generation_routes_explicit_credentials_through_seegen() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            request_body = json.loads(request.content)
            assert request.headers["Authorization"] == "Bearer test-key"
            assert request_body["model"] == "seedream-v4.0"
            assert request_body["watermark"] is False
            assert "n" not in request_body
            return httpx.Response(
                202,
                json={
                    "task_id": "img-public",
                    "status": "processing",
                    "model": "seedream-v4.0",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "task_id": "img-public",
                "status": "done",
                "image_urls": ["https://cdn.example.com/public.png"],
                "usage": {"generated_images": 1, "output_tokens": 100, "total_tokens": 120},
            },
            request=request,
        )

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(route)))

    response = litellm.image_generation(
        prompt="draw a lighthouse",
        model="seegen/seedream-v4.0",
        n=2,
        api_key="test-key",
        client=client,
        drop_params=True,
        timeout=1,
    )

    assert response.data is not None
    assert response.data[0].url == "https://cdn.example.com/public.png"


@pytest.mark.asyncio
async def test_aimage_generation_completes_seegen_queue_task() -> None:
    def route(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                202,
                json={
                    "task_id": "img-async",
                    "status": "processing",
                    "model": "nano-banana-2",
                    "created_at": "2026-09-13T12:00:00Z",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "task_id": "img-async",
                "status": "done",
                "image_urls": ["https://cdn.example.com/async.png"],
                "usage": {"generated_images": 1, "output_tokens": 100, "total_tokens": 120},
            },
            request=request,
        )

    client = AsyncHTTPHandler()
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(route))

    response = await litellm.aimage_generation(
        prompt="draw a lighthouse",
        model="seegen/nano-banana-2",
        size="2048x1536",
        api_key="test-key",
        client=client,
        timeout=1,
    )

    assert response.data is not None
    assert response.data[0].url == "https://cdn.example.com/async.png"
    await client.client.aclose()
