from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from typing_extensions import assert_never

from litellm.types.videos.main import VideoCreateOptionalRequestParams

from ..common_utils import EMPTY_JSON_OBJECT, JsonValue, SeeGenError, parse_json_mapping
from .models import SeeGenVideoFamily, model_name

STANDARD_PARAMS: Final = frozenset({"model", "prompt", "user", "extra_headers"})
_TUNING_PARAMS: Final = frozenset({"seconds", "size", "duration", "resolution", "ratio", "seed", "watermark"})
_HAPPYHORSE_CAPABILITIES: Final[Mapping[SeeGenVideoFamily, frozenset[str]]] = MappingProxyType(
    {
        SeeGenVideoFamily.HAPPYHORSE_T2V: frozenset(),
        SeeGenVideoFamily.HAPPYHORSE_I2V: frozenset({"image_url"}),
        SeeGenVideoFamily.HAPPYHORSE_R2V: frozenset({"input_reference"}),
        SeeGenVideoFamily.HAPPYHORSE_EDIT: frozenset({"input_reference", "video_urls", "base_video_url"}),
    }
)
_WAN_CAPABILITIES: Final = frozenset(
    {"image_url", "end_image_url", "input_reference", "video_urls", "audio_urls", "generate_audio"}
)
_SIZE_OPTIONS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "854x480": ("16:9", "480P"),
        "480x854": ("9:16", "480P"),
        "1280x720": ("16:9", "720P"),
        "720x1280": ("9:16", "720P"),
        "1920x1080": ("16:9", "1080P"),
        "1080x1920": ("9:16", "1080P"),
        "2560x1440": ("16:9", "2K"),
        "1440x2560": ("9:16", "2K"),
        "3840x2160": ("16:9", "4K"),
        "2160x3840": ("9:16", "4K"),
    }
)
_RESOLUTION_OPTIONS: Final[Mapping[str, str]] = MappingProxyType(
    {"480p": "480P", "720p": "720P", "1080p": "1080P", "2k": "2K", "4k": "4K"}
)


def capabilities(family: SeeGenVideoFamily) -> frozenset[str]:
    match family:
        case SeeGenVideoFamily.WAN:
            return _WAN_CAPABILITIES
        case (
            SeeGenVideoFamily.HAPPYHORSE_T2V
            | SeeGenVideoFamily.HAPPYHORSE_I2V
            | SeeGenVideoFamily.HAPPYHORSE_R2V
            | SeeGenVideoFamily.HAPPYHORSE_EDIT
        ):
            return _HAPPYHORSE_CAPABILITIES[family]
        case SeeGenVideoFamily.SEEDANCE:
            return frozenset()
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)


def supported_params(family: SeeGenVideoFamily) -> frozenset[str]:
    match family:
        case SeeGenVideoFamily.HAPPYHORSE_T2V:
            return _TUNING_PARAMS
        case SeeGenVideoFamily.HAPPYHORSE_I2V:
            return _TUNING_PARAMS | frozenset({"image_url"})
        case SeeGenVideoFamily.HAPPYHORSE_R2V:
            return _TUNING_PARAMS | frozenset({"input_reference"})
        case SeeGenVideoFamily.HAPPYHORSE_EDIT:
            return _TUNING_PARAMS | frozenset({"input_reference", "video_urls", "base_video_url", "audio_setting"})
        case SeeGenVideoFamily.WAN:
            return _TUNING_PARAMS | _WAN_CAPABILITIES | frozenset({"prompt_extend"})
        case SeeGenVideoFamily.SEEDANCE:
            return frozenset()
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)


def media_urls(value: JsonValue | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(item for item in value if isinstance(item, str) and item)
    raise SeeGenError(status_code=400, message=f"{name} must be a URL or list of URLs")


def happyhorse_edit_media(params: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    videos: Final = media_urls(params.get("video_urls", params.get("base_video_url")), "video_urls")
    if len(videos) != 1:
        raise SeeGenError(status_code=400, message="HappyHorse video-edit requires exactly one video reference")
    references: Final = media_urls(params.get("input_reference"), "input_reference")
    if len(references) > 5:
        raise SeeGenError(status_code=400, message="HappyHorse video-edit accepts at most 5 reference images")
    video: Final = parse_json_mapping(MappingProxyType({"type": "video", "url": videos[0]}))
    return (
        video,
        *(parse_json_mapping(MappingProxyType({"type": "reference_image", "url": url})) for url in references),
    )


def wan_media(params: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    first_frames: Final = media_urls(params.get("image_url"), "image_url")
    last_frames: Final = media_urls(params.get("end_image_url"), "end_image_url")
    references: Final = media_urls(params.get("input_reference"), "input_reference")
    videos: Final = media_urls(params.get("video_urls"), "video_urls")
    audios: Final = media_urls(params.get("audio_urls"), "audio_urls")
    if len(first_frames) > 1 or len(last_frames) > 1:
        raise SeeGenError(status_code=400, message="Wan accepts at most one first and last frame")
    if len(references) > 10 or len(videos) > 5 or len(audios) > 5:
        raise SeeGenError(status_code=400, message="Wan reference media exceeds the vendor limits")
    if (first_frames or last_frames) and (references or videos or audios):
        raise SeeGenError(status_code=400, message="Wan frame mode and reference mode are mutually exclusive")
    return tuple(
        parse_json_mapping(MappingProxyType({"type": media_type, "url": url}))
        for media_type, urls in (
            ("first_frame", first_frames),
            ("last_frame", last_frames),
            ("reference_image", references),
            ("reference_video", videos),
            ("reference_audio", audios),
        )
        for url in urls
    )


def request_media(
    params: Mapping[str, JsonValue],
    family: SeeGenVideoFamily,
    model: str,
) -> tuple[Mapping[str, JsonValue], ...]:
    match family:
        case SeeGenVideoFamily.HAPPYHORSE_T2V:
            return ()
        case SeeGenVideoFamily.HAPPYHORSE_I2V:
            first_frames: Final = media_urls(params.get("image_url"), "image_url")
            if len(first_frames) != 1:
                raise SeeGenError(status_code=400, message="HappyHorse i2v requires exactly one image_url")
            return (parse_json_mapping(MappingProxyType({"type": "first_frame", "url": first_frames[0]})),)
        case SeeGenVideoFamily.HAPPYHORSE_R2V:
            references: Final = media_urls(params.get("input_reference"), "input_reference")
            if not 1 <= len(references) <= 9:
                raise SeeGenError(status_code=400, message="HappyHorse r2v requires 1 to 9 reference images")
            return tuple(
                parse_json_mapping(MappingProxyType({"type": "reference_image", "url": url})) for url in references
            )
        case SeeGenVideoFamily.HAPPYHORSE_EDIT:
            return happyhorse_edit_media(params)
        case SeeGenVideoFamily.WAN:
            return wan_media(params)
        case SeeGenVideoFamily.SEEDANCE:
            raise SeeGenError(status_code=400, message=f"Seedance requires its Ark config: {model}")
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)


def _size_params(params: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    size: Final = params.get("size")
    if size is not None:
        if not isinstance(size, str) or size not in _SIZE_OPTIONS:
            raise SeeGenError(status_code=400, message=f"Unsupported SeeGen video size: {size}")
        ratio, resolution = _SIZE_OPTIONS[size]
        return MappingProxyType({"ratio": ratio, "resolution": resolution})
    ratio_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"ratio": params["ratio"]}) if "ratio" in params else EMPTY_JSON_OBJECT
    )
    resolution_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"resolution": _resolution(params["resolution"])})
        if "resolution" in params
        else EMPTY_JSON_OBJECT
    )
    return MappingProxyType({**ratio_params, **resolution_params})


def map_dashscope_params(
    params: VideoCreateOptionalRequestParams,
    family: SeeGenVideoFamily,
    model: str,
    drop_params: bool,
) -> Mapping[str, JsonValue]:
    supported: Final = supported_params(family)
    parsed: Final = parse_json_mapping(params)
    extra_body: Final = parsed.get("extra_body")
    if extra_body is not None and not isinstance(extra_body, dict):
        raise SeeGenError(status_code=400, message="extra_body must be an object")
    base_params: Final = MappingProxyType({key: value for key, value in parsed.items() if key != "extra_body"})
    extra_params: Final[Mapping[str, JsonValue]] = extra_body if isinstance(extra_body, dict) else EMPTY_JSON_OBJECT
    combined: Final = MappingProxyType({**base_params, **extra_params})
    unsupported: Final = tuple(key for key in combined if key not in supported and key not in STANDARD_PARAMS)
    if unsupported and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {unsupported}")
    selected: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {key: value for key, value in combined.items() if key in supported}
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
    mapped: Final[Mapping[str, JsonValue]] = MappingProxyType({**retained, **duration_params, **size_params})
    _validate_options(family, mapped, model)
    return mapped


def _duration(value: JsonValue | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise SeeGenError(status_code=400, message="SeeGen video duration must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise SeeGenError(status_code=400, message="SeeGen video duration must be an integer") from exc
    raise SeeGenError(status_code=400, message="SeeGen video duration must be an integer")


def _resolution(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise SeeGenError(status_code=400, message="resolution must be a string")
    normalized: Final = _RESOLUTION_OPTIONS.get(value.lower())
    if normalized is None:
        raise SeeGenError(status_code=400, message=f"Unsupported resolution: {value}")
    return normalized


def _validate_options(family: SeeGenVideoFamily, params: Mapping[str, JsonValue], model: str) -> None:
    duration: Final = params.get("duration")
    if duration is not None:
        duration_valid: Final = (
            isinstance(duration, int) and not isinstance(duration, bool) and (2 <= duration <= 30 or duration == -1)
            if family == SeeGenVideoFamily.WAN
            else isinstance(duration, int) and not isinstance(duration, bool) and 3 <= duration <= 15
        )
        if not duration_valid:
            raise SeeGenError(status_code=400, message=f"Invalid duration for {family.value}: {duration}")
    resolution: Final = params.get("resolution")
    allowed_resolutions: Final = (
        frozenset({"480P", "720P", "1080P", "2K", "4K"})
        if family == SeeGenVideoFamily.WAN
        or model_name(model) in frozenset({"happyhorse-1.1-t2v", "happyhorse-1.1-i2v", "happyhorse-1.1-r2v"})
        else frozenset({"720P", "1080P"})
        if family == SeeGenVideoFamily.HAPPYHORSE_EDIT
        else frozenset({"720P", "1080P", "2K", "4K"})
    )
    if resolution is not None and resolution not in allowed_resolutions:
        raise SeeGenError(status_code=400, message=f"Invalid resolution for {family.value}: {resolution}")
    ratio: Final = params.get("ratio")
    allowed_ratios: Final = frozenset({"adaptive", "16:9", "9:16", "1:1", "4:3", "3:4"})
    if ratio is not None and (ratio not in allowed_ratios or (family != SeeGenVideoFamily.WAN and ratio == "adaptive")):
        raise SeeGenError(status_code=400, message=f"Invalid ratio for {family.value}: {ratio}")
    seed: Final = params.get("seed")
    seed_valid: Final = (
        seed is None
        or isinstance(seed, int)
        and not isinstance(seed, bool)
        and (0 <= seed <= 2_147_483_647 or (family == SeeGenVideoFamily.WAN and seed == -1))
    )
    if not seed_valid:
        raise SeeGenError(status_code=400, message=f"Invalid seed for {family.value}: {seed}")
