import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.types.videos.main import VideoCreateOptionalRequestParams

from ..common_utils import EMPTY_JSON_OBJECT, JsonValue, SeeGenError, parse_json_mapping
from .models import SEEDANCE_25_MODELS, SEEDANCE_STANDARD_20_MODELS, model_name, video_family

SIZE_OPTIONS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
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
)
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


# Seedance's own output geometry per resolution tier and ratio (the sizes the
# platform pins per tier x aspect and sends as the OpenAI-standard `size`).
# SIZE_OPTIONS above lists the 16:9 / 9:16 pair per tier; the other four
# ratios Seedance serves (4:3, 3:4, 1:1, 21:9) share each tier's pixel
# budget, so a `size` is resolved by nearest ratio and nearest tier area
# rather than by an exact-string table that only knew two ratios (#82).
_RATIO_VALUES: Final[Mapping[str, float]] = MappingProxyType(
    {"16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0, "21:9": 21 / 9}
)
_TIER_AREAS: Final[Mapping[str, int]] = MappingProxyType(
    {"480p": 854 * 480, "720p": 1280 * 720, "1080p": 1920 * 1080, "2K": 2560 * 1440, "4K": 3840 * 2160}
)
# A size whose aspect is further than this (in log space, ~6%) from every
# ratio Seedance serves, or whose area sits between tiers, is refused rather
# than silently rounded to a geometry the caller did not ask for.
_RATIO_TOLERANCE: Final = 0.06
_AREA_TOLERANCE: Final = 0.12


def resolve_size(size: str) -> tuple[str, str]:
    """Map a WIDTHxHEIGHT `size` onto Seedance's (ratio, resolution) pair."""
    exact = SIZE_OPTIONS.get(size)
    if exact is not None:
        return exact
    width_text, separator, height_text = size.partition("x")
    if separator != "x" or not width_text.isdigit() or not height_text.isdigit():
        raise SeeGenError(status_code=400, message=f"Unsupported Seedance size: {size}")
    width, height = int(width_text), int(height_text)
    if width <= 0 or height <= 0:
        raise SeeGenError(status_code=400, message=f"Unsupported Seedance size: {size}")
    aspect: Final = math.log(width / height)
    ratio, ratio_distance = min(
        ((name, abs(aspect - math.log(value))) for name, value in _RATIO_VALUES.items()), key=lambda item: item[1]
    )
    area: Final = math.log(width * height)
    resolution, area_distance = min(
        ((name, abs(area - math.log(pixels))) for name, pixels in _TIER_AREAS.items()), key=lambda item: item[1]
    )
    if ratio_distance > _RATIO_TOLERANCE or area_distance > _AREA_TOLERANCE:
        raise SeeGenError(status_code=400, message=f"Unsupported Seedance size: {size}")
    return ratio, resolution


def _size_params(params: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    size: Final = params.get("size")
    if size is not None:
        if not isinstance(size, str):
            raise SeeGenError(status_code=400, message=f"Unsupported Seedance size: {size}")
        ratio, resolution = resolve_size(size)
        return MappingProxyType({"ratio": ratio, "resolution": resolution})
    ratio_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"ratio": params["ratio"]}) if "ratio" in params else EMPTY_JSON_OBJECT
    )
    resolution_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"resolution": params["resolution"]}) if "resolution" in params else EMPTY_JSON_OBJECT
    )
    return MappingProxyType({**ratio_params, **resolution_params})


def _edit_params(params: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if params.get("omni_reference_task_type") != "edit":
        return EMPTY_JSON_OBJECT
    requested_duration: Final = params.get("duration")
    if requested_duration not in (None, -1):
        raise SeeGenError(status_code=400, message="Seedance editing duration must be -1")
    return MappingProxyType({"duration": -1})


def map_seedance_params(
    params: VideoCreateOptionalRequestParams,
    model: str,
    drop_params: bool,
) -> Mapping[str, JsonValue]:
    normalized_model: Final = model_name(model)
    video_family(normalized_model)
    parsed: Final = parse_json_mapping(params)
    extra_body: Final = parsed.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise SeeGenError(status_code=400, message="extra_body must be an object")
    base_params: Final = MappingProxyType({key: value for key, value in parsed.items() if key != "extra_body"})
    extra_params: Final[Mapping[str, JsonValue]] = extra_body if isinstance(extra_body, dict) else EMPTY_JSON_OBJECT
    combined: Final = MappingProxyType({**base_params, **extra_params})
    unsupported: Final = tuple(
        key for key in combined if key not in SUPPORTED_PARAMS and key not in IGNORED_STANDARD_PARAMS
    )
    if unsupported and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {unsupported}")
    selected: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {key: value for key, value in combined.items() if key in SUPPORTED_PARAMS}
    )
    duration_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"duration": _duration(selected.get("seconds"))})
        if "seconds" in selected
        else MappingProxyType({"duration": _duration(selected.get("duration"))})
        if "duration" in selected
        else EMPTY_JSON_OBJECT
    )
    size_params: Final = _size_params(selected)
    transformed: Final = frozenset({"seconds", "duration", "size", "ratio", "resolution"})
    retained: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {key: value for key, value in selected.items() if key not in transformed}
    )
    normalized: Final[Mapping[str, JsonValue]] = MappingProxyType({**retained, **duration_params, **size_params})
    edit_params: Final = _edit_params(normalized)
    mapped: Final[Mapping[str, JsonValue]] = MappingProxyType({**normalized, **edit_params})
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
