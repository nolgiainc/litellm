from .path_transformation import (
    PATH_MODELS,
    PATH_TASK_KIND,
    KlingPathVideoConfig,
    is_kling_path_model,
)
from .transformation import KlingVideoConfig


def get_kling_video_config(model: str | None) -> KlingVideoConfig:
    """
    Pick the surface a Kling model actually lives on.

    Kling 3.0 Turbo, 3.0 Omni and O1 exist ONLY on the path-based API and the
    classic models exist only on /v1, so this is a hard routing decision and
    not a preference: sending either one to the other's transform produces a
    vendor-side rejection (1201/1203 on the classic surface, 401 code 1002 for
    the classic credential on the path surface).

    `model` is also the value a status or content lookup decodes out of a
    video id, which is why is_kling_path_model() answers for the encoded kind
    as well as for the model names.
    """
    return KlingPathVideoConfig() if is_kling_path_model(model) else KlingVideoConfig()


__all__ = [  # mutable-ok: module export list, the shape Python expects
    "PATH_MODELS",
    "PATH_TASK_KIND",
    "KlingPathVideoConfig",
    "KlingVideoConfig",
    "get_kling_video_config",
    "is_kling_path_model",
]
