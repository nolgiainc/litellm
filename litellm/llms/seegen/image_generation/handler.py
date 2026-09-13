from __future__ import annotations

from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, assert_never

import httpx

import litellm
from litellm.exceptions import Timeout
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # cached factory exposes legacy untyped params
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # cached factory exposes legacy untyped params
)
from litellm.types.utils import ImageResponse

from ..common_utils import (
    DEFAULT_MAX_POLLING_TIME,
    DEFAULT_POLLING_INTERVAL,
    EMPTY_HEADERS,
    EMPTY_JSON_OBJECT,
    AsyncHTTPClient,
    JsonValue,
    SeeGenError,
    SeeGenImageLogger,
    SyncHTTPClient,
    error_from_http_response,
    parse_headers,
    parse_json_mapping,
    parse_submitted_task,
)
from .parameters import SeeGenModelFamily, seegen_model_family
from .polling import SeeGenPoller, SeeGenPollRequest
from .response import as_b64_async, as_b64_sync
from .transformation import SeeGenImageGenerationConfig


@dataclass(frozen=True, slots=True)
class _SubmitRequest:
    model: str
    url: str
    headers: Mapping[str, str]
    data: Mapping[str, JsonValue]
    timeout: float | httpx.Timeout | None


@dataclass(frozen=True, slots=True)
class SeeGenImageGeneration:
    poll_interval: float = DEFAULT_POLLING_INTERVAL
    max_polling_time: float = DEFAULT_MAX_POLLING_TIME
    config: SeeGenImageGenerationConfig = field(default_factory=SeeGenImageGenerationConfig)

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        logging_obj: SeeGenImageLogger,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
    ) -> ImageResponse | Coroutine[None, None, ImageResponse]:
        if aimg_generation:
            return self.async_image_generation(
                model=model,
                prompt=prompt,
                model_response=model_response,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                extra_headers=extra_headers,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )
        sync_client: Final = client if isinstance(client, HTTPHandler) else _get_httpx_client()
        submit_request: Final = self._prepare_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        submitted: Final = parse_submitted_task(self._submit_sync(submit_request, sync_client))
        poll_request: Final = SeeGenPollRequest(
            url=f"{submit_request.url}/{submitted.task_id}",
            headers=submit_request.headers,
            timeout=submit_request.timeout,
        )
        poller: Final = SeeGenPoller(interval=self.poll_interval, max_wait=self.max_polling_time)
        final_response: Final = poller.poll_sync(poll_request, sync_client)
        result: Final = self.config.transform_image_generation_response(
            model=model,
            raw_response=final_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=parse_json_mapping(submit_request.data),
            optional_params=optional_params,
            litellm_params=EMPTY_JSON_OBJECT,
            encoding=None,
        )
        return (
            as_b64_sync(result, sync_client, timeout)
            if optional_params.get("response_format") == "b64_json"
            else result
        )

    async def async_image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        logging_obj: SeeGenImageLogger,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ImageResponse:
        async_client: Final = client or get_async_httpx_client(llm_provider=litellm.LlmProviders.SEEGEN)
        submit_request: Final = self._prepare_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
        )
        submitted: Final = parse_submitted_task(await self._submit_async(submit_request, async_client))
        poll_request: Final = SeeGenPollRequest(
            url=f"{submit_request.url}/{submitted.task_id}",
            headers=submit_request.headers,
            timeout=submit_request.timeout,
        )
        poller: Final = SeeGenPoller(interval=self.poll_interval, max_wait=self.max_polling_time)
        final_response: Final = await poller.poll_async(poll_request, async_client)
        result: Final = self.config.transform_image_generation_response(
            model=model,
            raw_response=final_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=parse_json_mapping(submit_request.data),
            optional_params=optional_params,
            litellm_params=EMPTY_JSON_OBJECT,
            encoding=None,
        )
        return (
            await as_b64_async(result, async_client, timeout)
            if optional_params.get("response_format") == "b64_json"
            else result
        )

    def _prepare_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        logging_obj: SeeGenImageLogger,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None,
    ) -> _SubmitRequest:
        api_key_value: Final = litellm_params.get("api_key")
        api_base_value: Final = litellm_params.get("api_base")
        drop_params_value: Final = litellm_params.get("drop_params")
        api_key: Final = api_key_value if isinstance(api_key_value, str) else None
        api_base: Final = api_base_value if isinstance(api_base_value, str) else None
        transform_litellm_params: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"drop_params": drop_params_value})
            if isinstance(drop_params_value, bool)
            else EMPTY_JSON_OBJECT
        )
        headers: Final = self.config.validate_environment(
            headers=extra_headers or EMPTY_HEADERS,
            model=model,
            messages=(),
            optional_params=optional_params,
            litellm_params=EMPTY_JSON_OBJECT,
            api_key=api_key,
        )
        url: Final = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=EMPTY_JSON_OBJECT,
        )
        data: Final = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=transform_litellm_params,
            headers=headers,
        )
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args=parse_json_mapping(
                MappingProxyType({"complete_input_dict": data, "api_base": url, "headers": headers})
            ),
        )
        return _SubmitRequest(model=model, url=url, headers=headers, data=data, timeout=timeout)

    def _post_sync(self, request: _SubmitRequest, client: SyncHTTPClient) -> httpx.Response:
        try:
            response: Final = client.post(
                url=request.url,
                headers=parse_headers(request.headers),
                json=parse_json_mapping(request.data),
                timeout=request.timeout,
            )
        except httpx.HTTPStatusError as exc:
            raise error_from_http_response(exc.response) from exc
        except httpx.HTTPError as exc:
            raise SeeGenError(status_code=502, message=f"SeeGen submit failed: {exc}") from exc
        if response.status_code != 202:
            raise SeeGenError(status_code=502, message=f"SeeGen submit returned HTTP {response.status_code}")
        return response

    async def _post_async(self, request: _SubmitRequest, client: AsyncHTTPClient) -> httpx.Response:
        try:
            response: Final = await client.post(
                url=request.url,
                headers=parse_headers(request.headers),
                json=parse_json_mapping(request.data),
                timeout=request.timeout,
            )
        except httpx.HTTPStatusError as exc:
            raise error_from_http_response(exc.response) from exc
        except httpx.HTTPError as exc:
            raise SeeGenError(status_code=502, message=f"SeeGen submit failed: {exc}") from exc
        if response.status_code != 202:
            raise SeeGenError(status_code=502, message=f"SeeGen submit returned HTTP {response.status_code}")
        return response

    def _submit_sync(self, request: _SubmitRequest, client: SyncHTTPClient) -> httpx.Response:
        try:
            return self._post_sync(request, client)
        except Timeout:
            family: Final = seegen_model_family(request.model)
            match family:
                case SeeGenModelFamily.GPT_IMAGE:
                    return self._post_sync(request, client)
                case SeeGenModelFamily.SEEDREAM | SeeGenModelFamily.NANO_BANANA:
                    raise
                case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
                    assert_never(unreachable)

    async def _submit_async(self, request: _SubmitRequest, client: AsyncHTTPClient) -> httpx.Response:
        try:
            return await self._post_async(request, client)
        except Timeout:
            family: Final = seegen_model_family(request.model)
            match family:
                case SeeGenModelFamily.GPT_IMAGE:
                    return await self._post_async(request, client)
                case SeeGenModelFamily.SEEDREAM | SeeGenModelFamily.NANO_BANANA:
                    raise
                case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
                    assert_never(unreachable)


seegen_image_generation: Final = SeeGenImageGeneration()
