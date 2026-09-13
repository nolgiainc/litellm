from __future__ import annotations

import base64
from typing import Final

import httpx

from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import AsyncHTTPClient, SeeGenError, SyncHTTPClient


def _fetch_b64_sync(url: str, client: SyncHTTPClient, timeout: float | httpx.Timeout | None) -> str:
    response: Final = client.get(url=url, timeout=timeout)
    if response.status_code >= 400:
        raise SeeGenError(status_code=response.status_code, message=f"SeeGen image download failed: {response.text}")
    return base64.b64encode(response.content).decode()


async def _fetch_b64_async(url: str, client: AsyncHTTPClient, timeout: float | httpx.Timeout | None) -> str:
    response: Final = await client.get(url=url, timeout=timeout)
    if response.status_code >= 400:
        raise SeeGenError(status_code=response.status_code, message=f"SeeGen image download failed: {response.text}")
    return base64.b64encode(response.content).decode()


def as_b64_sync(
    response: ImageResponse,
    client: SyncHTTPClient,
    timeout: float | httpx.Timeout | None,
) -> ImageResponse:
    urls: Final = tuple(image.url for image in (response.data or []) if image.url is not None)
    return ImageResponse(
        created=response.created,
        data=[ImageObject(b64_json=_fetch_b64_sync(url, client, timeout)) for url in urls],
        usage=response.usage,
    )


async def as_b64_async(
    response: ImageResponse,
    client: AsyncHTTPClient,
    timeout: float | httpx.Timeout | None,
) -> ImageResponse:
    urls: Final = tuple(image.url for image in (response.data or []) if image.url is not None)
    data: Final = [ImageObject(b64_json=await _fetch_b64_async(url, client, timeout)) for url in urls]
    return ImageResponse(
        created=response.created,
        data=data,
        usage=response.usage,
    )
