from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, assert_never

from pydantic import TypeAdapter

from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams

from ..common_utils import JsonValue, SeeGenError


class SeeGenModelFamily(StrEnum):
    SEEDREAM = "seedream"
    GPT_IMAGE = "gpt_image"
    NANO_BANANA = "nano_banana"


_SEEDREAM_MODELS: Final = frozenset({"seedream-v4.0", "seedream-v4.5", "seedream-v5.0-lite", "seedream-v5.0-pro"})
_GPT_IMAGE_MODELS: Final = frozenset({"gpt-image-2", "gpt-image-2.5-sunburst", "gpt-image-2.5-flare"})
_NANO_BANANA_MODELS: Final = frozenset({"nano-banana-2", "nano-banana-pro"})
_GPT_25_MODELS: Final = frozenset({"gpt-image-2.5-sunburst", "gpt-image-2.5-flare"})
_NANO_ASPECT_RATIOS: Final = ("1:1", "4:3", "3:2", "5:4", "3:4", "4:5", "16:9", "9:16", "21:9", "2:3")
_JSON_LIST_ADAPTER: Final = TypeAdapter(list[JsonValue])
_SEEDREAM_BASE_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "image",
    "size",
    "response_format",
    "watermark",
)
_GPT_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "image",
    "mask",
    "size",
    "quality",
    "n",
    "background",
    "moderation",
    "output_format",
    "output_compression",
    "input_fidelity",
    "user",
    "response_format",
    "stream",
    "partial_images",
)
_NANO_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "image",
    "images",
    "size",
    "resolution",
    "aspect_ratio",
    "response_format",
)


def seegen_model_name(model: str) -> str:
    return model.split("/", 1)[-1]


def seegen_model_family(model: str) -> SeeGenModelFamily:
    model_name: Final = seegen_model_name(model)
    if model_name in _SEEDREAM_MODELS:
        return SeeGenModelFamily.SEEDREAM
    if model_name in _GPT_IMAGE_MODELS:
        return SeeGenModelFamily.GPT_IMAGE
    if model_name in _NANO_BANANA_MODELS:
        return SeeGenModelFamily.NANO_BANANA
    raise SeeGenError(status_code=400, message=f"Unsupported SeeGen image model: {model_name}")


def supported_openai_params(model: str) -> tuple[OpenAIImageGenerationOptionalParams, ...]:
    model_name: Final = seegen_model_name(model)
    family: Final = seegen_model_family(model)
    match family:
        case SeeGenModelFamily.SEEDREAM:
            sequential: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
                ()
                if model_name == "seedream-v5.0-pro"
                else ("sequential_image_generation", "sequential_image_generation_options")
            )
            lite: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
                ("output_format",) if model_name == "seedream-v5.0-lite" else ()
            )
            pro: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
                ("layer_decomposition", "background") if model_name == "seedream-v5.0-pro" else ()
            )
            return (*_SEEDREAM_BASE_PARAMS, *sequential, *lite, *pro)
        case SeeGenModelFamily.GPT_IMAGE:
            return (
                _GPT_PARAMS
                if model_name in _GPT_25_MODELS
                else tuple(param for param in _GPT_PARAMS if param != "input_fidelity")
            )
        case SeeGenModelFamily.NANO_BANANA:
            return _NANO_PARAMS
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)


def _normalized_images(value: JsonValue, limit: int) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SeeGenError(status_code=400, message="image inputs must be a string or list of strings")
    if len(value) > limit:
        raise SeeGenError(status_code=400, message=f"image inputs exceed the model limit of {limit}")
    return tuple(item for item in value if isinstance(item, str))


def _nano_size(size: str) -> tuple[str, str | None]:
    if size in frozenset({"1K", "2K", "4K"}):
        return size, None
    try:
        width, height = (int(part) for part in size.lower().split("x"))
    except ValueError as exc:
        raise SeeGenError(status_code=400, message=f"Invalid Nano Banana size: {size}") from exc
    if width <= 0 or height <= 0:
        raise SeeGenError(status_code=400, message=f"Invalid Nano Banana size: {size}")
    max_edge: Final = max(width, height)
    resolution: Final = "1K" if max_edge <= 1024 else "2K" if max_edge <= 2048 else "4K"
    target: Final = width / height
    aspect_ratio: Final = min(
        _NANO_ASPECT_RATIOS,
        key=lambda ratio: abs((int(ratio.split(":")[0]) / int(ratio.split(":")[1])) - target),
    )
    return resolution, aspect_ratio


def _map_seedream(params: Mapping[str, JsonValue], model: str) -> Mapping[str, JsonValue]:
    image: Final = params.get("image")
    limit: Final = 10 if seegen_model_name(model) == "seedream-v5.0-pro" else 14
    normalized: Final = _normalized_images(image, limit) if image is not None else None
    mapped_image: Final[JsonValue | None] = (
        normalized[0]
        if normalized is not None and len(normalized) == 1
        else _JSON_LIST_ADAPTER.validate_python(normalized)
        if normalized is not None
        else None
    )
    image_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"image": mapped_image}) if mapped_image is not None else MappingProxyType({})
    )
    return MappingProxyType({**params, **image_params, "watermark": False})


def _map_gpt_image(
    params: Mapping[str, JsonValue],
    model: str,
    drop_params: bool,
) -> Mapping[str, JsonValue]:
    rejected: Final = (
        *(("stream",) if params.get("stream") not in (None, False) else ()),
        *(("partial_images",) if params.get("partial_images") is not None else ()),
    )
    quality: Final = params.get("quality")
    supported_qualities: Final = (
        frozenset({"auto", "low", "medium", "high", "xhigh", "max"})
        if seegen_model_name(model) in _GPT_25_MODELS
        else frozenset({"auto", "low", "medium", "high"})
    )
    quality_invalid: Final = quality is not None and (
        not isinstance(quality, str) or quality not in supported_qualities
    )
    n: Final = params.get("n")
    n_invalid: Final = n is not None and (not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 10)
    size: Final = params.get("size")
    size_invalid: Final = _gpt_size_invalid(size)
    response_format: Final = params.get("response_format")
    response_format_invalid: Final = response_format not in (None, "url", "b64_json")
    edit_only_invalid: Final = params.get("image") is None and any(
        params.get(key) is not None for key in ("input_fidelity", "mask")
    )
    invalid: Final = (
        *rejected,
        *(("quality",) if quality_invalid else ()),
        *(("n",) if n_invalid else ()),
        *(("size",) if size_invalid else ()),
        *(("response_format",) if response_format_invalid else ()),
        *(("input_fidelity",) if edit_only_invalid and params.get("input_fidelity") is not None else ()),
        *(("mask",) if edit_only_invalid and params.get("mask") is not None else ()),
    )
    if invalid and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {invalid}")
    ignored: Final = frozenset({"stream", "partial_images"})
    return MappingProxyType({key: value for key, value in params.items() if key not in invalid and key not in ignored})


def _gpt_size_invalid(size: JsonValue) -> bool:
    if size in (None, "auto"):
        return False
    if not isinstance(size, str):
        return True
    try:
        width, height = (int(part) for part in size.lower().split("x"))
    except ValueError:
        return True
    return (
        width <= 0
        or height <= 0
        or width % 16 != 0
        or height % 16 != 0
        or max(width, height) > 3840
        or width / height < 1 / 3
        or width / height > 3
    )


def _map_nano_banana(params: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    size: Final = params.get("size")
    resolution, aspect_ratio = _nano_size(size) if isinstance(size, str) else (params.get("resolution"), None)
    raw_images: Final = params.get("images", params.get("image"))
    images: Final = _normalized_images(raw_images, 10) if raw_images is not None else None
    serialized_images: Final = _JSON_LIST_ADAPTER.validate_python(images) if images is not None else None
    ignored: Final = frozenset({"size", "image", "images"})
    retained: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {key: value for key, value in params.items() if key not in ignored}
    )
    resolution_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"resolution": resolution}) if resolution is not None else MappingProxyType({})
    )
    ratio_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"aspect_ratio": aspect_ratio}) if aspect_ratio is not None else MappingProxyType({})
    )
    image_params: Final[Mapping[str, JsonValue]] = (
        MappingProxyType({"images": serialized_images}) if serialized_images is not None else MappingProxyType({})
    )
    return MappingProxyType({**retained, **resolution_params, **ratio_params, **image_params})


def _explicit_params(
    combined: Mapping[str, JsonValue],
    model: str,
    drop_params: bool,
) -> Mapping[str, JsonValue]:
    supported: Final = frozenset(supported_openai_params(model))
    unsupported: Final = tuple(key for key in combined if key not in supported)
    if unsupported and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported parameters for {model}: {unsupported}")
    response_format: Final = combined.get("response_format")
    response_format_invalid: Final = response_format not in (None, "url", "b64_json")
    if response_format_invalid and not drop_params:
        raise SeeGenError(status_code=400, message=f"Unsupported response_format for {model}: {response_format}")
    return MappingProxyType(
        {
            key: value
            for key, value in combined.items()
            if key in supported and not (key == "response_format" and response_format_invalid)
        }
    )


def map_openai_params(
    non_default_params: Mapping[str, JsonValue],
    optional_params: Mapping[str, JsonValue],
    model: str,
    drop_params: bool,
) -> Mapping[str, JsonValue]:
    combined: Final = MappingProxyType({**non_default_params, **optional_params})
    family: Final = seegen_model_family(model)
    match family:
        case SeeGenModelFamily.SEEDREAM:
            return _map_seedream(_explicit_params(combined, model, drop_params), model)
        case SeeGenModelFamily.GPT_IMAGE:
            return _map_gpt_image(combined, model, drop_params)
        case SeeGenModelFamily.NANO_BANANA:
            return _map_nano_banana(_explicit_params(combined, model, drop_params))
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)
