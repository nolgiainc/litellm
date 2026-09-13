from enum import StrEnum
from typing import Final

from ..common_utils import SeeGenError


class SeeGenVideoFamily(StrEnum):
    SEEDANCE = "seedance"
    HAPPYHORSE_T2V = "happyhorse_t2v"
    HAPPYHORSE_I2V = "happyhorse_i2v"
    HAPPYHORSE_R2V = "happyhorse_r2v"
    HAPPYHORSE_EDIT = "happyhorse_edit"
    WAN = "wan"


SEEDANCE_25_MODELS: Final = frozenset(
    {
        "doubao-seedance-2-5-260628",
        "dreamina-seedance-2-5-260628",
        "nsfw-seedance-2-5",
    }
)
SEEDANCE_STANDARD_20_MODELS: Final = frozenset(
    {
        "doubao-seedance-2-0-260128",
        "dreamina-seedance-2-0-260128",
        "nsfw-seedance-2-0",
    }
)
SEEDANCE_FAST_MINI_MODELS: Final = frozenset(
    {
        "doubao-seedance-2-0-fast-260128",
        "doubao-seedance-2-0-mini-260615",
        "dreamina-seedance-2-0-fast-260128",
        "dreamina-seedance-2-0-mini-260615",
        "nsfw-seedance-2-0-fast",
        "nsfw-seedance-2-0-mini",
    }
)
SEEDANCE_MODELS: Final = SEEDANCE_25_MODELS | SEEDANCE_STANDARD_20_MODELS | SEEDANCE_FAST_MINI_MODELS
HAPPYHORSE_MODELS: Final = frozenset(
    {
        "happyhorse-1.1-t2v",
        "happyhorse-1.1-i2v",
        "happyhorse-1.1-r2v",
        "happyhorse-1.0-t2v",
        "happyhorse-1.0-i2v",
        "happyhorse-1.0-r2v",
        "happyhorse-1.0-video-edit",
    }
)
WAN_MODELS: Final = frozenset(
    {
        "wan3.0-video",
        "wan3.0-video-prime",
        "nsfw-wan3.0-video",
        "nsfw-wan3.0-video-prime",
    }
)


def model_name(model: str) -> str:
    return model.split("/", 1)[-1]


def video_family(model: str) -> SeeGenVideoFamily:
    normalized = model_name(model)
    if normalized in SEEDANCE_MODELS:
        return SeeGenVideoFamily.SEEDANCE
    if normalized.endswith("-t2v") and normalized in HAPPYHORSE_MODELS:
        return SeeGenVideoFamily.HAPPYHORSE_T2V
    if normalized.endswith("-i2v") and normalized in HAPPYHORSE_MODELS:
        return SeeGenVideoFamily.HAPPYHORSE_I2V
    if normalized.endswith("-r2v") and normalized in HAPPYHORSE_MODELS:
        return SeeGenVideoFamily.HAPPYHORSE_R2V
    if normalized.endswith("-video-edit") and normalized in HAPPYHORSE_MODELS:
        return SeeGenVideoFamily.HAPPYHORSE_EDIT
    if normalized in WAN_MODELS:
        return SeeGenVideoFamily.WAN
    raise SeeGenError(status_code=400, message=f"Unsupported SeeGen video model: {normalized}")
