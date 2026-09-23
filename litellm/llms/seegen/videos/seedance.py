from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import (
    VideoCancelAccepted,
    VideoCancelPreflight,
    VideoCancelProceed,
    VideoCancelRefusal,
    VideoCancelRequest,
    VideoCancelVerdict,
    VideoCreateOptionalRequestParams,
    VideoObject,
)
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.videos.capabilities import CapabilityParamSupport, DeclaredCapabilityParams

from ..common_utils import EMPTY_JSON_OBJECT, JsonValue, SeeGenError, error_from_http_response, parse_json_mapping
from .base import SeeGenVideoConfig
from .frames import resolve_frame_media
from .models import model_name, video_family
from .seedance_parameters import (
    CAPABILITIES,
    IGNORED_STANDARD_PARAMS,
    SUPPORTED_PARAMS,
    map_seedance_params,
    media_urls,
)

_CREATE_PATH: Final = "/v1/contents/generations/tasks"
_STRING_LIST_ADAPTER: Final = TypeAdapter(list[str])
_JSON_LIST_ADAPTER: Final = TypeAdapter(list[JsonValue])
_EMPTY_REQUEST_FILES: Final = TypeAdapter(list[tuple[str, str]]).validate_python(())
_CANCEL_NOT_FOUND: Final = VideoCancelRefusal(reason="not_found", message="SeeGen has no Seedance task with this id")


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


class _TaskError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    code: str | None = None
    message: str | None = None


class _TaskResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
    content: tuple[_ContentItem, ...] | _FlatContent | None = None
    usage: _Usage | None = None
    message: str | None = None
    failure_reason: str | None = None
    error: str | _TaskError | None = None


def _failure_code(task: _TaskResponse) -> str:
    """The provider's own error code when Ark sent one, else the lifecycle status.

    Ark names WHY a task failed in ``error.code`` (for example
    ``InputImageSensitiveContentDetected.PrivacyInformation`` for a reference
    its checker refused, ``OutputVideoSensitiveContentDetected.PolicyViolation``
    for a clip it moderated after rendering). That code is the only stable
    discriminator between the two: the human sentence next to it changes
    wording between refusals. Surfacing ``task.status`` here instead threw
    it away, so the platform could not tell an input refusal from a render
    failure.
    """
    if isinstance(task.error, _TaskError) and task.error.code:
        return task.error.code
    return task.status


def _failure_message(task: _TaskResponse) -> str:
    if task.failure_reason:
        return task.failure_reason
    if task.message:
        return task.message
    if isinstance(task.error, str) and task.error:
        return task.error
    if isinstance(task.error, _TaskError):
        message: Final = task.error.message
        if message:
            return message
    return f"Seedance task ended with status {task.status}"


class SeeGenSeedanceVideoConfig(SeeGenVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig requires a list
        video_family(model)
        return _STRING_LIST_ADAPTER.validate_python(SUPPORTED_PARAMS | IGNORED_STANDARD_PARAMS)

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        video_family(model)
        return DeclaredCapabilityParams(CAPABILITIES)

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseVideoConfig requires a dict
        return parse_json_mapping(map_seedance_params(video_create_optional_params, model, drop_params))

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, JsonValue],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, JsonValue], RequestFiles, str]:  # mutable-ok: BaseVideoConfig requires a dict body
        params: Final = parse_json_mapping(video_create_optional_request_params)
        text_content: Final = parse_json_mapping(MappingProxyType({"type": "text", "text": prompt}))
        video_urls: Final = media_urls(params.get("video_urls"), "video_urls")
        audio_urls: Final = media_urls(params.get("audio_urls"), "audio_urls")
        frames: Final = resolve_frame_media(
            image_url=media_urls(params.get("image_url"), "first_frame"),
            end_image_url=media_urls(params.get("end_image_url"), "last_frame"),
            input_reference=media_urls(params.get("input_reference"), "input_reference"),
            image_urls=media_urls(params.get("image_urls"), "image_urls"),
            reference_media=video_urls + audio_urls,
        )
        image_roles: Final = (
            (frames.first_frames, "first_frame"),
            (frames.last_frames, "last_frame"),
            (frames.reference_images, "reference_image"),
        )
        image_content: Final = tuple(
            parse_json_mapping(
                MappingProxyType(
                    {
                        "type": "image_url",
                        "image_url": parse_json_mapping(MappingProxyType({"url": url})),
                        "role": role,
                    }
                )
            )
            for urls, role in image_roles
            for url in urls
        )
        video_content: Final = tuple(
            parse_json_mapping(
                MappingProxyType(
                    {
                        "type": "video_url",
                        "video_url": parse_json_mapping(MappingProxyType({"url": url})),
                        "role": "reference_video",
                    }
                )
            )
            for url in video_urls
        )
        audio_content: Final = tuple(
            parse_json_mapping(
                MappingProxyType(
                    {
                        "type": "audio_url",
                        "audio_url": parse_json_mapping(MappingProxyType({"url": url})),
                        "role": "reference_audio",
                    }
                )
            )
            for url in audio_urls
        )
        content: Final = _JSON_LIST_ADAPTER.validate_python(
            (text_content, *image_content, *video_content, *audio_content)
        )
        media_keys: Final = frozenset(
            {"image_url", "end_image_url", "input_reference", "image_urls", "video_urls", "audio_urls"}
        )
        forwarded: Final[Mapping[str, JsonValue]] = MappingProxyType(
            {
                key: value
                for key, value in params.items()
                if key not in IGNORED_STANDARD_PARAMS and key not in media_keys
            }
        )
        request_data: Final = parse_json_mapping(
            MappingProxyType({"model": model_name(model), "content": content, **forwarded})
        )
        return request_data, _EMPTY_REQUEST_FILES, f"{api_base.rstrip('/')}{_CREATE_PATH}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, JsonValue] | None = None,
    ) -> VideoObject:
        payload: Final = self._json_response(raw_response)
        try:
            submitted: Final = _SubmitResponse.model_validate(payload)
        except ValidationError as exc:
            raise SeeGenError(status_code=502, message=f"upstream_malformed: {exc}") from exc
        self._model = model_name(model)
        raw_status: Final = submitted.status
        status: Final = (
            "queued"
            if raw_status is None
            else "processing"
            if raw_status in frozenset({"queued", "running"})
            else raw_status
        )
        duration: Final = request_data.get("duration") if request_data is not None else None
        resolution: Final = request_data.get("resolution") if request_data is not None else None
        duration_usage: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"duration_seconds": float(duration)})
            if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0
            else EMPTY_JSON_OBJECT
        )
        resolution_usage: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"video_resolution": resolution.lower()})
            if isinstance(resolution, str)
            else EMPTY_JSON_OBJECT
        )
        usage: Final = parse_json_mapping(MappingProxyType({**duration_usage, **resolution_usage}))
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
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, JsonValue]]:  # mutable-ok: BaseVideoConfig requires a dict query
        task_id: Final = self._remember_video_model(video_id)
        return (
            f"{api_base.rstrip('/')}{_CREATE_PATH}/{self._encoded_task_id(task_id)}",
            parse_json_mapping(EMPTY_JSON_OBJECT),
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        task: Final = self._parse_task(raw_response)
        status: Final = (
            "processing"
            if task.status in frozenset({"queued", "running"})
            else "completed"
            if task.status == "succeeded"
            else "failed"
        )
        error: Final[Mapping[str, JsonValue] | None] = (
            MappingProxyType({"code": _failure_code(task), "message": _failure_message(task)})
            if status == "failed"
            else None
        )
        usage: Final = (
            parse_json_mapping(
                MappingProxyType(
                    {
                        "completion_tokens": task.usage.completion_tokens,
                        "total_tokens": task.usage.total_tokens,
                    }
                )
            )
            if task.usage is not None
            else parse_json_mapping(EMPTY_JSON_OBJECT)
        )
        video_id: Final = (
            encode_video_id_with_provider(task.id, custom_llm_provider, self._model) if custom_llm_provider else task.id
        )
        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            error=parse_json_mapping(error) if error is not None else None,
            usage=usage,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, JsonValue]]:  # mutable-ok: BaseVideoConfig requires a dict query
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

    def transform_video_cancel_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
    ) -> VideoCancelRequest:
        task_id: Final = self._remember_video_model(video_id)
        task_url: Final = f"{api_base.rstrip('/')}{_CREATE_PATH}/{self._encoded_task_id(task_id)}"
        return VideoCancelRequest(status_url=task_url, cancel_method="DELETE", cancel_url=task_url)

    def transform_video_cancel_status_response(self, raw_response: httpx.Response) -> VideoCancelPreflight:
        # Ark's DELETE cancels only a queued task and permanently deletes the record of a finished one,
        # so it is sent for the queued state alone.
        if raw_response.status_code == 404:
            return _CANCEL_NOT_FOUND
        task: Final = self._parse_task(raw_response)
        match task.status:
            case "queued":
                return VideoCancelProceed(VideoCancelAccepted(outcome="cancelled", provider_status=task.status))
            case "cancelled":
                return VideoCancelAccepted(outcome="cancelled", provider_status=task.status)
            case "running" | "succeeded" | "failed" | "expired":
                return VideoCancelRefusal(
                    reason="too_late",
                    message=f"Seedance task is {task.status} and can no longer be cancelled",
                )

    def transform_video_cancel_response(
        self,
        raw_response: httpx.Response,
        proceed: VideoCancelProceed,
    ) -> VideoCancelVerdict:
        if raw_response.is_success:
            return proceed.if_accepted
        if raw_response.status_code == 404:
            return _CANCEL_NOT_FOUND
        if raw_response.status_code == 409:
            return VideoCancelRefusal(
                reason="too_late",
                message="Seedance task left the queue before the cancel arrived",
            )
        raise error_from_http_response(raw_response)

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, JsonValue]]:  # mutable-ok: BaseVideoConfig requires a dict body
        task_id: Final = self._remember_video_model(video_id)
        self._requested_video_id = task_id
        return (
            f"{api_base.rstrip('/')}{_CREATE_PATH}/{self._encoded_task_id(task_id)}",
            parse_json_mapping(EMPTY_JSON_OBJECT),
        )

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
