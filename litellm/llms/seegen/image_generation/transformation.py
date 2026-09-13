from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import httpx
from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage, ImageUsageInputTokensDetails

from ..common_utils import (
    DEFAULT_API_BASE,
    IMAGE_GENERATION_PATH,
    JsonValue,
    SeeGenError,
    SeeGenGptUsage,
    SeeGenImageLogger,
    SeeGenUsage,
    error_from_response,
    parse_headers,
    parse_json_mapping,
    parse_polled_task,
)

_OPENAI_PARAMS_LIST_ADAPTER: Final = TypeAdapter(list[OpenAIImageGenerationOptionalParams])
_IMAGE_OBJECT_LIST_ADAPTER: Final = TypeAdapter(list[ImageObject])

if TYPE_CHECKING:
    import tiktoken
from .parameters import (
    SeeGenModelFamily,
    seegen_model_family,
    seegen_model_name,
    supported_openai_params,
)
from .parameters import map_openai_params as map_seegen_openai_params


def _image_usage(usage: SeeGenUsage | SeeGenGptUsage | None) -> ImageUsage:
    match usage:
        case SeeGenUsage():
            input_tokens: Final = max(0, usage.total_tokens - usage.output_tokens)
            return ImageUsage(
                input_tokens=input_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=input_tokens),
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
            )
        case SeeGenGptUsage():
            raw: Final = usage.raw_usage
            if raw is None:
                return ImageUsage(
                    input_tokens=0,
                    input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
                    output_tokens=0,
                    total_tokens=0,
                )
            details: Final = raw.input_tokens_details
            return ImageUsage(
                input_tokens=raw.input_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=details.image_tokens if details is not None else 0,
                    text_tokens=details.text_tokens if details is not None else raw.input_tokens,
                ),
                output_tokens=raw.output_tokens,
                total_tokens=raw.total_tokens,
            )
        case None:
            return ImageUsage(
                input_tokens=0,
                input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
                output_tokens=0,
                total_tokens=0,
            )
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)


class SeeGenImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig requires a list
        return _OPENAI_PARAMS_LIST_ADAPTER.validate_python(supported_openai_params(model))

    def map_openai_params(
        self,
        non_default_params: Mapping[str, JsonValue],
        optional_params: Mapping[str, JsonValue],
        model: str,
        drop_params: bool,
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseImageGenerationConfig requires a dict
        return parse_json_mapping(map_seegen_openai_params(non_default_params, optional_params, model, drop_params))

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseImageGenerationConfig requires concrete dict headers
        resolved_key: Final = api_key or get_secret_str("SEEGEN_API_KEY")
        if not resolved_key:
            raise SeeGenError(status_code=401, message="SEEGEN_API_KEY is not set")
        family: Final = seegen_model_family(model)
        match family:
            case SeeGenModelFamily.GPT_IMAGE:
                provided_key: Final = next(
                    (value for key, value in headers.items() if key.lower() == "idempotency-key"),
                    None,
                )
                request_headers: Final = MappingProxyType(
                    {key: value for key, value in headers.items() if key.lower() != "idempotency-key"}
                )
                return parse_headers(
                    MappingProxyType(
                        {
                            **request_headers,
                            "Authorization": f"Bearer {resolved_key}",
                            "Content-Type": "application/json",
                            "Idempotency-Key": provided_key or str(uuid4()),
                        }
                    )
                )
            case SeeGenModelFamily.SEEDREAM | SeeGenModelFamily.NANO_BANANA:
                return parse_headers(
                    MappingProxyType(
                        {**headers, "Authorization": f"Bearer {resolved_key}", "Content-Type": "application/json"}
                    )
                )
            case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
                assert_never(unreachable)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        stream: bool | None = None,
    ) -> str:
        base_url: Final = (api_base or get_secret_str("SEEGEN_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        return f"{base_url}{IMAGE_GENERATION_PATH}"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        headers: Mapping[str, str],
    ) -> dict[str, JsonValue]:  # mutable-ok: BaseImageGenerationConfig requires a dict
        model_name: Final = seegen_model_name(model)
        family: Final = seegen_model_family(model)
        drop_params: Final = litellm_params.get("drop_params") is True
        match family:
            case SeeGenModelFamily.SEEDREAM:
                supported: Final = frozenset(supported_openai_params(model))
                unsupported: Final = tuple(key for key in optional_params if key not in supported)
                if unsupported and not drop_params:
                    raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {unsupported}")
                ignored: Final = frozenset({"response_format", "watermark", "size"})
                forwarded: Final = MappingProxyType(
                    {key: value for key, value in optional_params.items() if key in supported and key not in ignored}
                )
                return parse_json_mapping(
                    MappingProxyType(
                        {
                            "model": model_name,
                            "prompt": prompt,
                            **forwarded,
                            "size": optional_params.get("size", "2048x2048"),
                            "response_format": "url",
                            "watermark": False,
                        }
                    )
                )
            case SeeGenModelFamily.GPT_IMAGE:
                gpt_ignored: Final = frozenset({"response_format", "stream", "partial_images"})
                gpt_forwarded: Final = MappingProxyType(
                    {key: value for key, value in optional_params.items() if key not in gpt_ignored}
                )
                return parse_json_mapping(
                    MappingProxyType({"model": model_name, "prompt": prompt, **gpt_forwarded, "response_format": "url"})
                )
            case SeeGenModelFamily.NANO_BANANA:
                nano_supported: Final = frozenset(supported_openai_params(model))
                nano_unsupported: Final = tuple(key for key in optional_params if key not in nano_supported)
                if nano_unsupported and not drop_params:
                    raise SeeGenError(
                        status_code=400, message=f"Unsupported parameters for {model}: {nano_unsupported}"
                    )
                nano_forwarded: Final = MappingProxyType(
                    {
                        key: value
                        for key, value in optional_params.items()
                        if key in nano_supported and key != "response_format"
                    }
                )
                return parse_json_mapping(MappingProxyType({"model": model_name, "prompt": prompt, **nano_forwarded}))
            case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
                assert_never(unreachable)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: SeeGenImageLogger,
        request_data: Mapping[str, JsonValue],
        optional_params: Mapping[str, JsonValue],
        litellm_params: Mapping[str, JsonValue],
        encoding: tiktoken.Encoding | None,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        task: Final = parse_polled_task(raw_response)
        if task.status != "done" or not task.image_urls:
            raise error_from_response(
                status_code=502,
                payload=MappingProxyType(
                    {"error": "invalid_image_response", "message": task.failure_reason or "missing image URLs"}
                ),
                headers=raw_response.headers,
            )
        return ImageResponse(
            data=_IMAGE_OBJECT_LIST_ADAPTER.validate_python(tuple(ImageObject(url=url) for url in task.image_urls)),
            usage=_image_usage(task.usage),
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> SeeGenError:
        return SeeGenError(status_code=status_code, message=error_message, headers=parse_headers(headers))
