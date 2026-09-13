from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Never

import httpx
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # cached factory exposes legacy untyped params
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # cached factory exposes legacy untyped params
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider

from ..common_utils import (
    DEFAULT_API_BASE,
    AsyncHTTPClient,
    JsonValue,
    SeeGenError,
    SyncHTTPClient,
    error_from_http_response,
    error_from_response,
    parse_headers,
)

_JSON_MAPPING_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


class SeeGenVideoConfig(BaseVideoConfig):
    def __init__(
        self,
        model: str | None = None,
        sync_client: SyncHTTPClient | None = None,
        async_client: AsyncHTTPClient | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._sync_client = sync_client
        self._async_client = async_client
        self._requested_video_id: str | None = None

    def set_status_lookup_client(self, client: HTTPHandler | AsyncHTTPHandler) -> None:
        if isinstance(client, AsyncHTTPHandler):
            self._async_client = client
        else:
            self._sync_client = client

    def _http_client(self) -> SyncHTTPClient:
        return self._sync_client or _get_httpx_client()

    def _async_http_client(self) -> AsyncHTTPClient:
        return self._async_client or get_async_httpx_client(llm_provider=litellm.LlmProviders.SEEGEN)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig requires concrete dict headers
        params_api_key: Final = litellm_params.api_key if litellm_params is not None else None
        resolved_key: Final = api_key or params_api_key or get_secret_str("SEEGEN_API_KEY")
        if not resolved_key:
            raise SeeGenError(status_code=401, message="SEEGEN_API_KEY is not set")
        return parse_headers(
            MappingProxyType(
                {
                    **headers,
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                }
            )
        )

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, JsonValue],
    ) -> str:
        return (api_base or get_secret_str("SEEGEN_API_BASE") or DEFAULT_API_BASE).rstrip("/")

    def _json_response(self, response: httpx.Response) -> Mapping[str, JsonValue]:
        if not response.is_success:
            raise error_from_http_response(response)
        try:
            return MappingProxyType(_JSON_MAPPING_ADAPTER.validate_json(response.content))
        except ValidationError as exc:
            raise SeeGenError(
                status_code=502,
                message="upstream_malformed: SeeGen returned an invalid JSON object",
                headers=response.headers,
                response=response,
            ) from exc

    def _remember_video_model(self, video_id: str) -> str:
        decoded: Final = decode_video_id_with_provider(video_id)
        decoded_model: Final = decoded.get("model_id")
        if decoded_model:
            self._model = decoded_model
        return decoded.get("video_id", video_id)

    @staticmethod
    def _encoded_task_id(task_id: str) -> str:
        return encode_url_path_segment(task_id, field_name="video_id")

    def _extract_video_url(self, payload: Mapping[str, JsonValue]) -> str:
        raise NotImplementedError

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url: Final = self._extract_video_url(self._json_response(raw_response))
        video_response: Final = self._http_client().get(url=video_url)
        if not video_response.is_success:
            raise error_from_http_response(video_response)
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url: Final = self._extract_video_url(self._json_response(raw_response))
        video_response: Final = await self._async_http_client().get(url=video_url)
        if not video_response.is_success:
            raise error_from_http_response(video_response)
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_body: Mapping[str, JsonValue] | None = None,
    ) -> Never:
        raise NotImplementedError("Video remix is not supported by SeeGen")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by SeeGen")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, JsonValue] | None = None,
    ) -> Never:
        raise NotImplementedError("Video listing is not supported by SeeGen")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> Never:
        raise NotImplementedError("Video listing is not supported by SeeGen")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, JsonValue]]:  # mutable-ok: BaseVideoConfig requires a dict body
        raise NotImplementedError("Video cancellation is not supported for this SeeGen family")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video cancellation is not supported for this SeeGen family")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> SeeGenError:
        try:
            payload: Final = _JSON_MAPPING_ADAPTER.validate_json(error_message)
        except ValidationError:
            return SeeGenError(status_code=status_code, message=error_message, headers=parse_headers(headers))
        return error_from_response(status_code, payload, headers)
