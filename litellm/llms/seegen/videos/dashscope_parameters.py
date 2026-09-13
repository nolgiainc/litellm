from collections.abc import Mapping
from typing import Final, assert_never

from litellm.types.videos.main import VideoCreateOptionalRequestParams

from ..common_utils import JsonValue, SeeGenError, parse_json_mapping
from .models import SeeGenVideoFamily

STANDARD_PARAMS: Final = frozenset({"model", "prompt", "user", "extra_headers"})
_TUNING_PARAMS: Final = frozenset({"seconds", "size", "duration", "resolution", "ratio", "seed", "watermark"})
_HAPPYHORSE_CAPABILITIES: Final[dict[SeeGenVideoFamily, frozenset[str]]] = {
    SeeGenVideoFamily.HAPPYHORSE_T2V: frozenset(),
    SeeGenVideoFamily.HAPPYHORSE_I2V: frozenset({"image_url"}),
    SeeGenVideoFamily.HAPPYHORSE_R2V: frozenset({"input_reference"}),
    SeeGenVideoFamily.HAPPYHORSE_EDIT: frozenset({"input_reference", "video_urls", "base_video_url"}),
}
_WAN_CAPABILITIES: Final = frozenset(
    {"image_url", "end_image_url", "input_reference", "video_urls", "audio_urls", "generate_audio"}
)
_SIZE_OPTIONS: Final[dict[str, tuple[str, str]]] = {
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
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]
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
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]
            assert_never(unreachable)


def media_urls(value: JsonValue | None, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(item for item in value if isinstance(item, str) and item)
    raise SeeGenError(status_code=400, message=f"{name} must be a URL or list of URLs")


def add_happyhorse_edit_media(
    params: dict[str, JsonValue],
    media: list[JsonValue],
) -> None:
    videos: Final = media_urls(params.pop("video_urls", params.pop("base_video_url", None)), "video_urls")
    if len(videos) != 1:
        raise SeeGenError(status_code=400, message="HappyHorse video-edit requires exactly one video reference")
    references: Final = media_urls(params.pop("input_reference", None), "input_reference")
    if len(references) > 5:
        raise SeeGenError(status_code=400, message="HappyHorse video-edit accepts at most 5 reference images")
    media.append({"type": "video", "url": videos[0]})
    media.extend({"type": "reference_image", "url": url} for url in references)


def add_wan_media(params: dict[str, JsonValue], media: list[JsonValue]) -> None:
    first_frames: Final = media_urls(params.pop("image_url", None), "image_url")
    last_frames: Final = media_urls(params.pop("end_image_url", None), "end_image_url")
    references: Final = media_urls(params.pop("input_reference", None), "input_reference")
    videos: Final = media_urls(params.pop("video_urls", None), "video_urls")
    audios: Final = media_urls(params.pop("audio_urls", None), "audio_urls")
    if len(first_frames) > 1 or len(last_frames) > 1:
        raise SeeGenError(status_code=400, message="Wan accepts at most one first and last frame")
    if len(references) > 10 or len(videos) > 5 or len(audios) > 5:
        raise SeeGenError(status_code=400, message="Wan reference media exceeds the vendor limits")
    if (first_frames or last_frames) and (references or videos or audios):
        raise SeeGenError(status_code=400, message="Wan frame mode and reference mode are mutually exclusive")
    media.extend({"type": "first_frame", "url": url} for url in first_frames)
    media.extend({"type": "last_frame", "url": url} for url in last_frames)
    media.extend({"type": "reference_image", "url": url} for url in references)
    media.extend({"type": "reference_video", "url": url} for url in videos)
    media.extend({"type": "reference_audio", "url": url} for url in audios)


def map_dashscope_params(
    params: VideoCreateOptionalRequestParams,
    family: SeeGenVideoFamily,
    model: str,
    drop_params: bool,
) -> dict[str, JsonValue]:
    supported: Final = supported_params(family)
    parsed: Final = parse_json_mapping(params)
    extra_body: Final = parsed.pop("extra_body", None)
    if extra_body is not None and not isinstance(extra_body, dict):
        raise SeeGenError(status_code=400, message="extra_body must be an object")
    combined: Final = {**parsed, **(extra_body or {})}
    unsupported: Final = tuple(key for key in combined if key not in supported and key not in STANDARD_PARAMS)
    if unsupported and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {unsupported}")
    mapped: Final[dict[str, JsonValue]] = {key: value for key, value in combined.items() if key in supported}
    if "seconds" in mapped:
        mapped["duration"] = _duration(mapped.pop("seconds"))
    elif "duration" in mapped:
        mapped["duration"] = _duration(mapped["duration"])
    size: Final = mapped.pop("size", None)
    if size is not None:
        if not isinstance(size, str) or size not in _SIZE_OPTIONS:
            raise SeeGenError(status_code=400, message=f"Unsupported SeeGen video size: {size}")
        mapped["ratio"], mapped["resolution"] = _SIZE_OPTIONS[size]
    elif "resolution" in mapped:
        mapped["resolution"] = _resolution(mapped["resolution"])
    _validate_options(family, mapped)
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
    normalized: Final = {
        "480p": "480P",
        "720p": "720P",
        "1080p": "1080P",
        "2k": "2K",
        "4k": "4K",
    }.get(value.lower())
    if normalized is None:
        raise SeeGenError(status_code=400, message=f"Unsupported resolution: {value}")
    return normalized


def _validate_options(family: SeeGenVideoFamily, params: Mapping[str, JsonValue]) -> None:
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
