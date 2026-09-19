import asyncio
import copy
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, Final
from unittest.mock import AsyncMock, Mock

import httpx
import orjson
import pytest
from fastapi import FastAPI
from pydantic import JsonValue, TypeAdapter
from starlette.requests import Request
from starlette.responses import Response

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints


@pytest.mark.asyncio
async def test_image_generation_prompt_rerouting(monkeypatch):
    """Ensure image prompts are exposed to guardrails and restored afterwards."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: Dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        modified = {
            **data,
            "messages": [
                {
                    "role": "user",
                    "content": "sanitized prompt",
                }
            ],
        }
        return modified

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    async def fake_post_call_success_hook(*, data, user_api_key_dict, response):
        return response

    async def fake_post_call_response_headers_hook(**kwargs):
        return {"x-callback-test": "value"}

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
        post_call_success_hook=fake_post_call_success_hook,
        post_call_response_headers_hook=fake_post_call_response_headers_hook,
    )

    captured_route_request_data: Dict[str, Any] = {}

    async def fake_route_request(*, data, **kwargs):  # type: ignore[override]
        captured_route_request_data.update(data)

        async def _inner():
            class FakeResponse(dict):
                _hidden_params = {}

            return FakeResponse(result="ok")

        return _inner()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/generations",
        "headers": [],
    }
    body = orjson.dumps({"prompt": "original prompt"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr(
        "litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request
    )

    result = await endpoints.image_generation(
        request=request,
        fastapi_response=response,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert pre_call_input["messages"][0]["content"] == "original prompt"
    assert captured_route_request_data["prompt"] == "sanitized prompt"
    assert "messages" not in captured_route_request_data
    assert response.headers.get("x-callback-test") == "value"


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_null", (False, True))
@pytest.mark.parametrize("background_removal", (False, True))
async def test_promptless_image_generation_over_http(
    monkeypatch: pytest.MonkeyPatch, explicit_null: bool, background_removal: bool,
) -> None:
    from litellm.proxy import proxy_server

    provider_model: Final = "fal_ai/fal-ai/bria/background/remove" if background_removal else "openai/gpt-image-2"
    router: Final = litellm.Router(
        model_list=[  # mutable-ok: Router requires a mutable deployment list.
            {  # mutable-ok: Router consumes mutable deployment dictionaries.
                "model_name": "image-model",
                "litellm_params": {"model": provider_model, "api_key": "test-key"},  # mutable-ok: Provider parameters.
            }
        ],
        num_retries=0,
    )

    def provider_response(request: httpx.Request) -> httpx.Response:
        assert background_removal
        assert request.method == "POST"
        assert str(request.url) == "https://fal.run/fal-ai/bria/background/remove"
        body: Final = TypeAdapter(Mapping[str, JsonValue]).validate_json(request.content)
        assert body.get("image_url") == "https://example.com/input.png"
        assert "prompt" not in body
        return httpx.Response(200, content=b'{"image":{"url":"https://example.com/output.png"}}')

    async def add_request_data(
        data: Mapping[str, object], request: Request, general_settings: Mapping[str, object],
        user_api_key_dict: UserAPIKeyAuth, version: str, proxy_config: object,
    ) -> Mapping[str, object]:
        return data

    async def post_success(
        data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth, response: litellm.ImageResponse,
    ) -> litellm.ImageResponse:
        return response

    def authenticated_user() -> UserAPIKeyAuth:
        return UserAPIKeyAuth()

    app: Final = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.user_api_key_auth] = authenticated_user
    app.add_exception_handler(ProxyException, proxy_server.openai_exception_handler)
    provider_transport: Final = Mock(side_effect=provider_response)
    with httpx.Client(transport=httpx.MockTransport(provider_transport)) as transport:
        provider_client: Final = HTTPHandler(client=transport)

        async def pre_call(
            user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, object], call_type: str,
        ) -> dict[str, object]:  # mutable-ok: The endpoint pops and augments request fields after the hook.
            return {**data, "client": provider_client}  # mutable-ok: Inject transport through the request hook boundary.

        hooks: Final = SimpleNamespace(
            pre_call_hook=AsyncMock(side_effect=pre_call),
            post_call_success_hook=AsyncMock(side_effect=post_success),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value=None),
            update_request_status=AsyncMock(),
        )
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(proxy_server, "user_model", None)
        monkeypatch.setattr(proxy_server, "general_settings", MappingProxyType({}))
        monkeypatch.setattr(proxy_server, "shared_aiohttp_session", None)
        monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", add_request_data)
        monkeypatch.setattr(proxy_server, "proxy_logging_obj", hooks)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response: Final = await client.post(
                "/v1/images/generations",
                json={  # mutable-ok: httpx serializes a JSON request dictionary.
                    "model": "image-model",
                    "image_url": "https://example.com/input.png",
                    **MappingProxyType({"prompt": None} if explicit_null else {}),
                },
            )

    if background_removal:
        assert response.status_code == 200, response.text
        assert response.json()["data"][0]["url"] == "https://example.com/output.png"
        provider_transport.assert_called_once()
    else:
        assert response.status_code == 400, response.text
        assert "requires a prompt" in response.json()["error"]["message"]
        provider_transport.assert_not_called()
