from collections.abc import Mapping
from typing import Final

from litellm.types.videos.main import VideoCreateOptionalRequestParams

from ..common_utils import JsonValue, SeeGenError, parse_json_mapping
from .models import SEEDANCE_25_MODELS, SEEDANCE_STANDARD_20_MODELS, model_name, video_family

SIZE_OPTIONS: Final[dict[str, tuple[str, str]]] = {
    "854x480": ("16:9", "480p"),
    "480x854": ("9:16", "480p"),
    "1280x720": ("16:9", "720p"),
    "720x1280": ("9:16", "720p"),
    "1920x1080": ("16:9", "1080p"),
    "1080x1920": ("9:16", "1080p"),
    "2560x1440": ("16:9", "2K"),
    "1440x2560": ("9:16", "2K"),
    "3840x2160": ("16:9", "4K"),
    "2160x3840": ("9:16", "4K"),
}
SUPPORTED_PARAMS: Final = frozenset(
    {
        "seconds",
        "size",
        "image_url",
        "end_image_url",
        "input_reference",
        "image_urls",
        "video_urls",
        "audio_urls",
        "generate_audio",
        "ratio",
        "duration",
        "resolution",
        "bitrate_mode",
        "watermark",
        "output_format",
        "omni_reference_task_type",
        "return_last_frame",
        "safety_identifier",
    }
)
IGNORED_STANDARD_PARAMS: Final = frozenset({"model", "prompt", "user", "extra_headers"})
CAPABILITIES: Final = frozenset(
    {
        "image_url",
        "end_image_url",
        "input_reference",
        "image_urls",
        "video_urls",
        "audio_urls",
        "bitrate_mode",
        "generate_audio",
    }
)
_RATIOS: Final = frozenset({"adaptive", "16:9", "9:16", "4:3", "3:4", "1:1", "21:9"})


def media_urls(value: JsonValue | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(item for item in value if isinstance(item, str) and item)
    raise SeeGenError(status_code=400, message=f"{name} must be a URL or list of URLs")


def map_seedance_params(
    params: VideoCreateOptionalRequestParams,
    model: str,
    drop_params: bool,
) -> dict[str, JsonValue]:
    normalized_model: Final = model_name(model)
    video_family(normalized_model)
    parsed: Final = parse_json_mapping(params)
    extra_body: Final = parsed.pop("extra_body", None)
    if extra_body is not None and not isinstance(extra_body, dict):
        raise SeeGenError(status_code=400, message="extra_body must be an object")
    combined: Final = {**parsed, **(extra_body or {})}
    unsupported: Final = tuple(
        key for key in combined if key not in SUPPORTED_PARAMS and key not in IGNORED_STANDARD_PARAMS
    )
    if unsupported and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {unsupported}")
    mapped: Final[dict[str, JsonValue]] = {key: value for key, value in combined.items() if key in SUPPORTED_PARAMS}
    if "seconds" in mapped:
        mapped["duration"] = _duration(mapped.pop("seconds"))
    elif "duration" in mapped:
        mapped["duration"] = _duration(mapped["duration"])
    size: Final = mapped.pop("size", None)
    if size is not None:
        if not isinstance(size, str) or size not in SIZE_OPTIONS:
            raise SeeGenError(status_code=400, message=f"Unsupported Seedance size: {size}")
        mapped["ratio"], mapped["resolution"] = SIZE_OPTIONS[size]
    if mapped.get("omni_reference_task_type") == "edit":
        requested_duration: Final = mapped.get("duration")
        if requested_duration not in (None, -1):
            raise SeeGenError(status_code=400, message="Seedance editing duration must be -1")
        mapped["duration"] = -1
    _validate_options(normalized_model, mapped)
    return mapped


def _duration(value: JsonValue | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SeeGenError(status_code=400, message="Seedance duration must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise SeeGenError(status_code=400, message="Seedance duration must be an integer") from exc
    raise SeeGenError(status_code=400, message="Seedance duration must be an integer")


def _validate_options(model: str, params: Mapping[str, JsonValue]) -> None:
    duration: Final = params.get("duration")
    if duration is not None:
        maximum: Final = 30 if model in SEEDANCE_25_MODELS else 15
        if (
            not isinstance(duration, int)
            or isinstance(duration, bool)
            or (duration != -1 and not 4 <= duration <= maximum)
        ):
            raise SeeGenError(status_code=400, message=f"Invalid duration for {model}: {duration}")
    ratio: Final = params.get("ratio")
    if ratio is not None and ratio not in _RATIOS:
        raise SeeGenError(status_code=400, message=f"Invalid ratio for {model}: {ratio}")
    resolution: Final = params.get("resolution")
    allowed_resolutions: Final = (
        frozenset({"480p", "720p", "1080p", "2K", "4K"})
        if model in SEEDANCE_25_MODELS
        else frozenset({"480p", "720p", "1080p", "4K"})
        if model in SEEDANCE_STANDARD_20_MODELS
        else frozenset({"480p", "720p"})
    )
    if resolution is not None and resolution not in allowed_resolutions:
        raise SeeGenError(status_code=400, message=f"Invalid resolution for {model}: {resolution}")
    if params.get("output_format") is not None and model not in SEEDANCE_25_MODELS:
        raise SeeGenError(status_code=400, message=f"output_format is only supported by Seedance 2.5: {model}")
