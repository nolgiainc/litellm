from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.videos.capabilities import CapabilityParamSupport, DeclaredCapabilityParams

from ..common_utils import EMPTY_HEADERS, EMPTY_JSON_OBJECT, JsonValue, SeeGenError, parse_headers, parse_json_mapping
from .base import SeeGenVideoConfig
from .dashscope_parameters import (
    STANDARD_PARAMS,
    capabilities,
    map_dashscope_params,
    request_media,
    supported_params,
)
from .models import SeeGenVideoFamily, model_name, video_family

_CREATE_PATH: Final = "/api/v1/services/aigc/video-generation/video-synthesis"
_TASK_PATH: Final = "/api/v1/tasks"
_STRING_LIST_ADAPTER: Final = TypeAdapter(list[str])
_JSON_LIST_ADAPTER: Final = TypeAdapter(list[JsonValue])
_EMPTY_REQUEST_FILES: Final = TypeAdapter(list[tuple[str, str]]).validate_python(())


class _DashOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    task_id: str
    task_status: str
    video_url: str | None = None
    message: str | None = None


class _DashUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    output_video_duration: float | None = Field(default=None, ge=0)
    duration: float | None = Field(default=None, ge=0)
    SR: int | str | None = None


class _DashResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    request_id: str | None = None
    output: _DashOutput
    usage: _DashUsage | None = None


class SeeGenDashScopeVideoConfig(SeeGenVideoConfig):
    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig requires concrete dict headers
        resolved_headers: Final = super().validate_environment(headers, model, api_key, litellm_params)
        effective_model: Final = model or self._model
        async_header: Final[Mapping[str, str]] = (
            MappingProxyType({"X-DashScope-Async": "enable"})
            if effective_model and video_family(effective_model) != SeeGenVideoFamily.WAN
            else EMPTY_HEADERS
        )
        return parse_headers(MappingProxyType({**resolved_headers, **async_header}))

    def supports_promptless_video_create(self, model: str) -> bool:
        return video_family(model) == SeeGenVideoFamily.WAN

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig requires a list
        return _STRING_LIST_ADAPTER.validate_python(supported_params(video_family(model)) | STANDARD_PARAMS)

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        family: Final = video_family(model)
        if family == SeeGenVideoFamily.SEEDANCE:
            raise SeeGenError(status_code=400, message=f"Seedance requires its Ark config: {model}")
        return DeclaredCapabilityParams(capabilities(family))

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseVideoConfig requires a dict
        family: Final = video_family(model)
        return parse_json_mapping(map_dashscope_params(video_create_optional_params, family, model, drop_params))

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, JsonValue],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, JsonValue], RequestFiles, str]:  # mutable-ok: BaseVideoConfig requires a dict body
        family: Final = video_family(model)
        params: Final = parse_json_mapping(video_create_optional_request_params)
        media_objects: Final = request_media(params, family, model)
        if family == SeeGenVideoFamily.WAN and not prompt and not media_objects:
            raise SeeGenError(status_code=400, message="Wan requires a prompt or media input")
        media_params: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"media": _JSON_LIST_ADAPTER.validate_python(media_objects)})
            if media_objects
            else EMPTY_JSON_OBJECT
        )
        input_data: Final = parse_json_mapping(MappingProxyType({"prompt": prompt, **media_params}))
        parameter_names: Final = frozenset(
            {
                "duration",
                "resolution",
                "ratio",
                "seed",
                "watermark",
                "audio_setting",
                "prompt_extend",
            }
        )
        base_parameters: Final[Mapping[str, JsonValue]] = MappingProxyType(
            {key: value for key, value in params.items() if key in parameter_names}
        )
        provider_parameters: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"audio": params["generate_audio"]})
            if family == SeeGenVideoFamily.WAN and "generate_audio" in params
            else EMPTY_JSON_OBJECT
        )
        parameters: Final = parse_json_mapping(
            MappingProxyType({**base_parameters, **provider_parameters, "watermark": False})
        )
        request_data: Final = parse_json_mapping(
            MappingProxyType({"model": model_name(model), "input": input_data, "parameters": parameters})
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
        response: Final = self._parse_response(raw_response)
        self._model = model_name(model)
        status, error = self._status_and_error(response.output)
        parameters_value: Final = request_data.get("parameters") if request_data is not None else None
        parameters: Final[Mapping[str, JsonValue]] = (
            parameters_value if isinstance(parameters_value, dict) else EMPTY_JSON_OBJECT
        )
        duration: Final = parameters.get("duration")
        resolution: Final = parameters.get("resolution")
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
            encode_video_id_with_provider(response.output.task_id, custom_llm_provider, self._model)
            if custom_llm_provider
            else response.output.task_id
        )
        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            model=self._model,
            error=parse_json_mapping(error) if error is not None else None,
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
            f"{api_base.rstrip('/')}{_TASK_PATH}/{self._encoded_task_id(task_id)}",
            parse_json_mapping(EMPTY_JSON_OBJECT),
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response: Final = self._parse_response(raw_response)
        status, error = self._status_and_error(response.output)
        duration: Final = (
            response.usage.output_video_duration or response.usage.duration if response.usage is not None else None
        )
        sr: Final = response.usage.SR if response.usage is not None else None
        duration_usage: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"duration_seconds": duration}) if duration is not None else EMPTY_JSON_OBJECT
        )
        resolution_usage: Final[Mapping[str, JsonValue]] = (
            MappingProxyType({"video_resolution": f"{sr}p" if isinstance(sr, int) else sr.lower()})
            if sr is not None
            else EMPTY_JSON_OBJECT
        )
        usage: Final = parse_json_mapping(MappingProxyType({**duration_usage, **resolution_usage}))
        video_id: Final = (
            encode_video_id_with_provider(response.output.task_id, custom_llm_provider, self._model)
            if custom_llm_provider
            else response.output.task_id
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
            response: Final = _DashResponse.model_validate(payload)
        except ValidationError as exc:
            raise SeeGenError(status_code=502, message=f"upstream_malformed: {exc}") from exc
        if response.output.task_status != "SUCCEEDED" or not response.output.video_url:
            _, error = self._status_and_error(response.output)
            message: Final = error["message"] if error is not None else "SeeGen video is still processing"
            raise SeeGenError(status_code=400, message=message)
        return response.output.video_url

    @staticmethod
    def _status_and_error(output: _DashOutput) -> tuple[str, Mapping[str, str] | None]:
        match output.task_status:
            case "PENDING" | "RUNNING":
                return "processing", None
            case "SUCCEEDED":
                return "completed", None
            case "FAILED":
                return "failed", MappingProxyType(
                    {"code": "failed", "message": output.message or "SeeGen video generation failed"}
                )
            case "CANCELED":
                return "failed", MappingProxyType(
                    {"code": "canceled", "message": output.message or "SeeGen video generation was canceled"}
                )
            case "UNKNOWN":
                return "failed", MappingProxyType(
                    {"code": "unknown", "message": output.message or "SeeGen task expired or was not found"}
                )
            case unknown:
                return "failed", MappingProxyType(
                    {
                        "code": "unknown_status",
                        "message": output.message or f"SeeGen returned unknown task status {unknown}",
                    }
                )

    def _parse_response(self, raw_response: httpx.Response) -> _DashResponse:
        try:
            return _DashResponse.model_validate(self._json_response(raw_response))
        except ValidationError as exc:
            raise SeeGenError(status_code=502, message=f"upstream_malformed: {exc}") from exc
