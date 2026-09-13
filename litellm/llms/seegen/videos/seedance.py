from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.videos.capabilities import CapabilityParamSupport, DeclaredCapabilityParams

from ..common_utils import JsonValue, SeeGenError, parse_json_mapping
from .base import SeeGenVideoConfig
from .models import model_name, video_family
from .seedance_parameters import (
    CAPABILITIES,
    IGNORED_STANDARD_PARAMS,
    SUPPORTED_PARAMS,
    map_seedance_params,
    media_urls,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_CREATE_PATH: Final = "/v1/contents/generations/tasks"


class _VideoURL(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    url: str


class _ContentItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    type: str
    video_url: _VideoURL | None = None


class _FlatContent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    video_url: str


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class _SubmitResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    status: str | None = None


class _TaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
    content: tuple[_ContentItem, ...] | _FlatContent | None = None
    usage: _Usage | None = None
    message: str | None = None
    failure_reason: str | None = None
    error: str | dict[str, JsonValue] | None = None


def _failure_message(task: _TaskResponse) -> str:
    if task.failure_reason:
        return task.failure_reason
    if task.message:
        return task.message
    if isinstance(task.error, str) and task.error:
        return task.error
    if isinstance(task.error, dict):
        message = task.error.get("message")
        if isinstance(message, str) and message:
            return message
    return f"Seedance task ended with status {task.status}"


class SeeGenSeedanceVideoConfig(SeeGenVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        video_family(model)
        return list(SUPPORTED_PARAMS | IGNORED_STANDARD_PARAMS)

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        video_family(model)
        return DeclaredCapabilityParams(CAPABILITIES)

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, JsonValue]:
        return map_seedance_params(video_create_optional_params, model, drop_params)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, JsonValue],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[dict[str, JsonValue], RequestFiles, str]:
        params: Final = parse_json_mapping(video_create_optional_request_params)
        content: Final[list[JsonValue]] = [{"type": "text", "text": prompt}]
        image_roles: Final = (
            (params.pop("image_url", None), "first_frame"),
            (params.pop("end_image_url", None), "last_frame"),
        )
        for value, role in image_roles:
            for url in media_urls(value, role):
                content.append({"type": "image_url", "image_url": {"url": url}, "role": role})
        for key in ("input_reference", "image_urls"):
            for url in media_urls(params.pop(key, None), key):
                content.append({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"})
        for url in media_urls(params.pop("video_urls", None), "video_urls"):
            content.append({"type": "video_url", "video_url": {"url": url}, "role": "reference_video"})
        for url in media_urls(params.pop("audio_urls", None), "audio_urls"):
            content.append({"type": "audio_url", "audio_url": {"url": url}, "role": "reference_audio"})
        request_data: Final = parse_json_mapping(
            {
                "model": model_name(model),
                "content": content,
                **{key: value for key, value in params.items() if key not in IGNORED_STANDARD_PARAMS},
            }
        )
        return request_data, [], f"{api_base.rstrip('/')}{_CREATE_PATH}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict[str, JsonValue] | None = None,
    ) -> VideoObject:
        payload: Final = self._json_response(raw_response)
        try:
            submitted: Final = _SubmitResponse.model_validate(payload)
        except ValidationError as exc:
            raise SeeGenError(status_code=502, message=f"upstream_malformed: {exc}") from exc
        self._model = model_name(model)
        raw_status: Final = submitted.status
        status: Final = (
            "queued" if raw_status is None else "processing" if raw_status in {"queued", "running"} else raw_status
        )
        usage: Final[dict[str, float | str]] = {}
        if request_data is not None:
            duration = request_data.get("duration")
            resolution = request_data.get("resolution")
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
                usage["duration_seconds"] = float(duration)
            if isinstance(resolution, str):
                usage["video_resolution"] = resolution.lower()
        video_id: Final = (
            encode_video_id_with_provider(submitted.id, custom_llm_provider, self._model)
            if custom_llm_provider
            else submitted.id
        )
        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            model=self._model,
            usage=usage,
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, JsonValue]]:
        task_id: Final = self._remember_video_model(video_id)
        return f"{api_base.rstrip('/')}{_CREATE_PATH}/{self._encoded_task_id(task_id)}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        task: Final = self._parse_task(raw_response)
        status: Final = (
            "processing"
            if task.status in {"queued", "running"}
            else "completed"
            if task.status == "succeeded"
            else "failed"
        )
        error: Final = {"code": task.status, "message": _failure_message(task)} if status == "failed" else None
        usage: Final = (
            {
                "completion_tokens": task.usage.completion_tokens,
                "total_tokens": task.usage.total_tokens,
            }
            if task.usage is not None
            else {}
        )
        video_id: Final = (
            encode_video_id_with_provider(task.id, custom_llm_provider, self._model) if custom_llm_provider else task.id
        )
        return VideoObject(id=video_id, object="video", status=status, error=error, usage=usage)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, JsonValue]]:
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def _extract_video_url(self, payload: Mapping[str, JsonValue]) -> str:
        try:
            task: Final = _TaskResponse.model_validate(payload)
        except ValidationError as exc:
            raise SeeGenError(status_code=502, message=f"upstream_malformed: {exc}") from exc
        if task.status != "succeeded":
            raise SeeGenError(status_code=400, message=_failure_message(task))
        match task.content:
            case _FlatContent(video_url=url):
                return url
            case tuple() as items:
                for item in items:
                    if item.type == "video_url" and item.video_url is not None:
                        return item.video_url.url
            case None:
                pass
        raise SeeGenError(status_code=502, message="upstream_malformed: Seedance response has no video URL")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],
    ) -> tuple[str, dict[str, JsonValue]]:
        task_id: Final = self._remember_video_model(video_id)
        self._requested_video_id = task_id
        return f"{api_base.rstrip('/')}{_CREATE_PATH}/{self._encoded_task_id(task_id)}", {}

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        return VideoObject(id=self._requested_video_id or "", object="video", status="cancelled")

    def _parse_task(self, response: httpx.Response) -> _TaskResponse:
        try:
            return _TaskResponse.model_validate(self._json_response(response))
        except ValidationError as exc:
            raise SeeGenError(status_code=502, message=f"upstream_malformed: {exc}") from exc
