from __future__ import annotations

import base64
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import AsyncHTTPClient, SeeGenError, SyncHTTPClient

_IMAGE_LIST_ADAPTER: Final = TypeAdapter(list[ImageObject])


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


async def _fetch_b64_objects(
    urls: tuple[str, ...],
    client: AsyncHTTPClient,
    timeout: float | httpx.Timeout | None,
) -> tuple[ImageObject, ...]:
    if not urls:
        return ()
    first: Final = ImageObject(b64_json=await _fetch_b64_async(urls[0], client, timeout))
    remaining: Final = await _fetch_b64_objects(urls[1:], client, timeout)
    return (first, *remaining)


def as_b64_sync(
    response: ImageResponse,
    client: SyncHTTPClient,
    timeout: float | httpx.Timeout | None,
) -> ImageResponse:
    urls: Final = tuple(image.url for image in (response.data or ()) if image.url is not None)
    return ImageResponse(
        created=response.created,
        data=_IMAGE_LIST_ADAPTER.validate_python(
            tuple(ImageObject(b64_json=_fetch_b64_sync(url, client, timeout)) for url in urls)
        ),
        usage=response.usage,
    )


async def as_b64_async(
    response: ImageResponse,
    client: AsyncHTTPClient,
    timeout: float | httpx.Timeout | None,
) -> ImageResponse:
    urls: Final = tuple(image.url for image in (response.data or ()) if image.url is not None)
    data: Final = _IMAGE_LIST_ADAPTER.validate_python(await _fetch_b64_objects(urls, client, timeout))
    return ImageResponse(
        created=response.created,
        data=data,
        usage=response.usage,
    )
