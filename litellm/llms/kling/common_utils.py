from typing import Final

from litellm.constants import KLING_DEFAULT_API_BASE
from litellm.secret_managers.main import get_secret_str

KLING_TASK_STATUS_MAP = {
    "submitted": "queued",
    "processing": "in_progress",
    "succeed": "completed",
    "succeeded": "completed",
    "failed": "failed",
}


def resolve_kling_api_base(api_base: str | None) -> str:
    base = api_base or get_secret_str("KLING_API_BASE") or KLING_DEFAULT_API_BASE
    return base.rstrip("/")


def strip_kling_prefix(model: str) -> str:
    stripped = model.removeprefix("kling/").strip("/")
    if not stripped:
        raise ValueError("Kling model id is empty after stripping the provider prefix")
    return stripped


def resolve_kling_path_api_base(api_base: str | None) -> str:
    """
    Base URL for Kling's PATH-BASED surface, which is the classic host without
    the `/v1` prefix: the endpoint is the model itself
    (POST /text-to-video/kling-3.0-turbo), not a versioned resource.

    Derived from the classic base rather than duplicated so a host override
    (KLING_API_BASE, a regional endpoint) reaches both surfaces from one place.
    """
    classic: Final = resolve_kling_api_base(api_base)
    return classic.removesuffix("/v1").rstrip("/")
