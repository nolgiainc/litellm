import time
from typing import Final

import jwt

import litellm
from litellm.secret_managers.main import get_secret_str

_KLING_JWT_TTL_SECONDS = 1800
_KLING_JWT_LEEWAY_SECONDS = 5


def resolve_kling_api_key(api_key: str | None) -> str:
    resolved = api_key or litellm.api_key or get_secret_str("KLING_API_KEY")
    if not resolved:
        raise ValueError(
            "Kling API key is required. Set KLING_API_KEY (format 'AccessKey:SecretKey') "
            "environment variable or pass the api_key parameter."
        )
    return resolved


def split_access_secret(api_key: str) -> tuple[str, str]:
    access_key, separator, secret_key = api_key.partition(":")
    if not separator or not access_key or not secret_key:
        raise ValueError("KLING_API_KEY must be in the form 'AccessKey:SecretKey' (two colon-separated parts).")
    return access_key, secret_key


def generate_kling_jwt(api_key: str) -> str:
    access_key, secret_key = split_access_secret(api_key)
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + _KLING_JWT_TTL_SECONDS,
        "nbf": now - _KLING_JWT_LEEWAY_SECONDS,
    }
    return jwt.encode(
        payload,
        secret_key,
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )


def kling_auth_headers(api_key: str | None) -> dict:
    token = generate_kling_jwt(resolve_kling_api_key(api_key))
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def resolve_kling_console_api_key(api_key: str | None) -> str:
    """
    The console key for Kling's newer PATH-BASED surface.

    Deliberately does NOT fall back to `litellm.api_key` the way
    resolve_kling_api_key() does. The two Kling credentials are different
    shapes serving different surfaces - an `AccessKey:SecretKey` pair that the
    classic /v1 routes sign short-lived JWTs with, against a single opaque
    `api-` token the path-based routes take as a plain bearer - and the generic
    fallback would happily hand the AK/SK pair to the path surface, which
    answers it with `401 code 1002` pointing at the console. A named env var or
    an explicit api_key, and nothing else.
    """
    resolved: Final = api_key or get_secret_str("KLING_CONSOLE_API_KEY")
    if not resolved:
        raise ValueError(
            "Kling console API key is required for the path-based Kling models "
            "(3.0 Turbo, 3.0 Omni, O1). Set KLING_CONSOLE_API_KEY (a single token minted at "
            "https://kling.ai/dev/api-key) or pass the api_key parameter. The classic "
            "KLING_API_KEY AccessKey:SecretKey pair is rejected by this surface."
        )
    if ":" in resolved:
        raise ValueError(
            "KLING_CONSOLE_API_KEY looks like an 'AccessKey:SecretKey' pair. The path-based Kling "
            "surface rejects AK/SK credentials (401 code 1002); mint a console key at "
            "https://kling.ai/dev/api-key."
        )
    return resolved


def kling_console_auth_headers(
    api_key: str | None,
) -> dict:  # mutable-ok: validate_environment merges into a mutable header dict
    return {  # mutable-ok: see the return annotation
        "Authorization": f"Bearer {resolve_kling_console_api_key(api_key)}",
        "Content-Type": "application/json",
    }
