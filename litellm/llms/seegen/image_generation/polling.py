from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, assert_never

import anyio
import httpx

from ..common_utils import (
    AsyncHTTPClient,
    SeeGenError,
    SyncHTTPClient,
    error_from_http_response,
    parse_polled_task,
)


@dataclass(frozen=True, slots=True)
class SeeGenPollRequest:
    url: str
    headers: Mapping[str, str]
    timeout: float | httpx.Timeout | None


@dataclass(frozen=True, slots=True)
class SeeGenPoller:
    interval: float
    max_wait: float

    def poll_sync(self, request: SeeGenPollRequest, client: SyncHTTPClient) -> httpx.Response:
        deadline: Final = time.monotonic() + self.max_wait
        for response in self._sync_responses(request, client, deadline):
            if (result := self._result_or_error(response)) is not None:
                return result
        raise SeeGenError(status_code=408, message=f"SeeGen polling timed out after {self.max_wait} seconds")

    async def poll_async(self, request: SeeGenPollRequest, client: AsyncHTTPClient) -> httpx.Response:
        deadline: Final = time.monotonic() + self.max_wait
        async for response in self._async_responses(request, client, deadline):
            if (result := self._result_or_error(response)) is not None:
                return result
        raise SeeGenError(status_code=408, message=f"SeeGen polling timed out after {self.max_wait} seconds")

    def _sync_responses(
        self,
        request: SeeGenPollRequest,
        client: SyncHTTPClient,
        deadline: float,
    ) -> Iterator[httpx.Response]:
        while time.monotonic() < deadline:
            yield client.get(url=request.url, headers=dict(request.headers), timeout=request.timeout)
            time.sleep(self.interval)

    async def _async_responses(
        self,
        request: SeeGenPollRequest,
        client: AsyncHTTPClient,
        deadline: float,
    ) -> AsyncIterator[httpx.Response]:
        while time.monotonic() < deadline:
            yield await client.get(url=request.url, headers=dict(request.headers), timeout=request.timeout)
            await anyio.sleep(self.interval)

    @staticmethod
    def _result_or_error(response: httpx.Response) -> httpx.Response | None:
        should_retry_values: Final = response.headers.get_list("x-should-retry")
        is_terminal: Final = any(value.lower() == "false" for value in should_retry_values)
        if response.status_code >= 400:
            response_error: Final = error_from_http_response(response)
            if is_terminal:
                raise SeeGenError(
                    status_code=400,
                    message=response_error.message,
                    headers=response.headers,
                    response=response,
                )
            raise response_error
        task: Final = parse_polled_task(response)
        if is_terminal:
            raise SeeGenError(
                status_code=400,
                message=task.failure_reason or "SeeGen marked the task terminal",
                headers=response.headers,
                response=response,
            )
        match task.status:
            case "done":
                return response
            case "failed":
                raise SeeGenError(status_code=400, message=task.failure_reason or "SeeGen image generation failed")
            case "processing":
                return None
            case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
                assert_never(unreachable)
