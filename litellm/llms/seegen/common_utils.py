from collections.abc import Mapping
from typing import Final, Literal, Protocol, TypeVar, assert_never

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException

_JsonInputT = TypeVar("_JsonInputT")

DEFAULT_API_BASE: Final = "https://api.seegen.ai"
IMAGE_GENERATION_PATH: Final = "/v1/images/generations"
DEFAULT_POLLING_INTERVAL: Final = 4.0
DEFAULT_MAX_POLLING_TIME: Final = 600.0


class SeeGenError(BaseLLMException):
    pass


class SyncHTTPClient(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response: ...

    def post(
        self,
        url: str,
        *,
        json: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response: ...


class AsyncHTTPClient(Protocol):
    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response: ...

    async def post(
        self,
        url: str,
        *,
        json: dict[str, JsonValue],
        headers: dict[str, str] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response: ...


class SeeGenGatewayError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: str
    message: str


class SeeGenOfficialErrorDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    param: str | None = None
    type: str | None = None


class SeeGenOfficialError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: SeeGenOfficialErrorDetail


class SeeGenUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    generated_images: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class SeeGenSubmittedTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(pattern=r"^img-[A-Za-z0-9_-]+$")
    status: Literal["processing"]
    model: str
    created_at: str


class SeeGenPolledTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str = Field(pattern=r"^img-[A-Za-z0-9_-]+$")
    status: Literal["processing", "done", "failed"]
    image_urls: tuple[str, ...] = ()
    usage: SeeGenUsage | None = None
    failure_reason: str | None = None


_ERROR_ADAPTER: Final[TypeAdapter[SeeGenGatewayError | SeeGenOfficialError]] = TypeAdapter(
    SeeGenGatewayError | SeeGenOfficialError
)
_JSON_MAPPING_ADAPTER: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])


def parse_json_mapping(value: Mapping[str, _JsonInputT]) -> dict[str, JsonValue]:
    return _JSON_MAPPING_ADAPTER.validate_python(value)


def _error_message(error: SeeGenGatewayError | SeeGenOfficialError) -> str:
    match error:
        case SeeGenGatewayError(error=code, message=message):
            return f"{code}: {message}"
        case SeeGenOfficialError(error=detail):
            param: Final = f" ({detail.param})" if detail.param else ""
            return f"{detail.code}{param}: {detail.message}"
        case unreachable:  # pyright: ignore[reportUnnecessaryComparison]  # exhaustive variant sentinel
            assert_never(unreachable)


def error_from_response(
    status_code: int,
    payload: Mapping[str, JsonValue],
    headers: Mapping[str, str] | httpx.Headers,
) -> SeeGenError:
    try:
        parsed: Final = _ERROR_ADAPTER.validate_python(dict(payload))
        return SeeGenError(
            status_code=status_code,
            message=_error_message(parsed),
            headers=dict(headers),
            body=dict(payload),
        )
    except ValidationError:
        return SeeGenError(
            status_code=status_code,
            message=str(dict(payload)),
            headers=dict(headers),
            body=dict(payload),
        )


def error_from_http_response(response: httpx.Response) -> SeeGenError:
    try:
        parsed: Final = _ERROR_ADAPTER.validate_json(response.content)
        return SeeGenError(
            status_code=response.status_code,
            message=_error_message(parsed),
            headers=response.headers,
            response=response,
        )
    except ValidationError:
        return SeeGenError(
            status_code=response.status_code,
            message=response.text or "SeeGen returned an invalid error response",
            headers=response.headers,
            response=response,
        )


def parse_submitted_task(response: httpx.Response) -> SeeGenSubmittedTask:
    try:
        return SeeGenSubmittedTask.model_validate_json(response.content)
    except ValidationError as exc:
        raise SeeGenError(
            status_code=502,
            message=f"Invalid SeeGen submit response: {exc}",
            headers=response.headers,
            response=response,
        ) from exc


def parse_polled_task(response: httpx.Response) -> SeeGenPolledTask:
    try:
        return SeeGenPolledTask.model_validate_json(response.content)
    except ValidationError as exc:
        raise SeeGenError(
            status_code=502,
            message=f"Invalid SeeGen poll response: {exc}",
            headers=response.headers,
            response=response,
        ) from exc
