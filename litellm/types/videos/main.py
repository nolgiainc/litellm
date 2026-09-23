import builtins
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from openai.types.audio.transcription_create_params import FileTypes
from pydantic import BaseModel, ConfigDict, PrivateAttr
from typing_extensions import ReadOnly, TypedDict


class VideoObject(BaseModel):
    """Represents a generated video object."""

    id: str
    object: Literal["video"]
    status: str
    created_at: int | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    error: dict[str, Any] | None = None
    progress: int | None = None
    remixed_from_video_id: str | None = None
    seconds: str | None = None
    size: str | None = None
    model: str | None = None
    usage: dict[str, Any] | None = None
    _hidden_params: dict[str, Any] = {}

    def __contains__(self, key) -> bool:
        # Define custom behavior for the 'in' operator
        return hasattr(self, key)

    def get(self, key, default=None):
        # Custom .get() method to access attributes with a default value if the attribute doesn't exist
        return getattr(self, key, default)

    def __getitem__(self, key):
        # Allow dictionary-style access to attributes
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except AttributeError:
            # if using pydantic v1
            return self.dict()


class VideoResponse(BaseModel):
    """Response object for video generation requests."""

    data: list[VideoObject]
    hidden_params: dict[str, Any] = {}

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except AttributeError:
            return self.dict()


class VideoCreateOptionalRequestParams(TypedDict, total=False):
    """
    TypedDict for Optional parameters supported by OpenAI's video creation API.

    Params here: https://platform.openai.com/docs/api-reference/videos/create
    """

    input_reference: FileTypes | None  # File reference for input image
    image: Any | None  # Image for image-to-video; dict with gcsUri/bytesBase64Encoded, or file-like object
    parameters: dict[str, Any] | None  # Provider-specific parameters block passed directly to the API
    model: str | None
    resolution: ReadOnly[str | None]
    seconds: str | None
    size: str | None
    characters: list[dict[str, str]] | None
    user: str | None
    extra_headers: dict[str, str] | None
    extra_body: dict[str, str] | None


class VideoCreateRequestParams(VideoCreateOptionalRequestParams, total=False):
    """
    TypedDict for request parameters supported by OpenAI's video creation API.

    Params here: https://platform.openai.com/docs/api-reference/videos/create
    """

    prompt: str


class DecodedVideoId(TypedDict, total=False):
    """Structure representing a decoded video ID"""

    custom_llm_provider: str | None
    model_id: str | None
    video_id: str


class CharacterObject(BaseModel):
    """Represents a character created from a video."""

    id: str
    object: Literal["character"] = "character"
    created_at: int
    name: str
    _hidden_params: dict[str, Any] = {}

    def __contains__(self, key) -> bool:
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def json(self, **kwargs):
        try:
            return self.model_dump(**kwargs)
        except AttributeError:
            return self.dict()


class VideoEditRequestParams(TypedDict, total=False):
    """TypedDict for video edit request parameters."""

    prompt: str
    video: dict[str, str]  # {"id": "video_123"}


class VideoExtensionRequestParams(TypedDict, total=False):
    """TypedDict for video extension request parameters."""

    prompt: str
    seconds: str
    video: dict[str, str]  # {"id": "video_123"}


VideoCancelOutcome: TypeAlias = Literal["cancelled", "requested", "partial"]
"""What an accepted cancel means for billing.

cancelled: the provider stopped the task before it started processing, so it is not billed.
requested: the provider accepted a stop signal for a task that was already processing; it may still
finish and bill.
partial: the provider stopped a task mid-render and bills it pro rata by ``progress``.
"""

VideoCancelRefusalReason: TypeAlias = Literal["too_late", "not_found", "unsupported"]


class VideoCancelObject(BaseModel):
    """A cancel the provider accepted. ``progress`` (0..1) is set only for a ``partial`` outcome."""

    model_config = ConfigDict(frozen=True)

    id: str
    object: Literal["video"] = "video"
    status: Literal["cancelled"] = "cancelled"
    cancel_outcome: VideoCancelOutcome
    provider_status: str
    progress: float | None = None
    _hidden_params: dict[str, builtins.object] = PrivateAttr(default_factory=dict)  # mutable-ok: call metadata


class VideoCancelRefusal(BaseModel):
    """A cancel the provider did not accept. Returned as a value so the router neither retries nor cools down on it."""

    model_config = ConfigDict(frozen=True)

    reason: VideoCancelRefusalReason
    message: str
    _hidden_params: dict[str, object] = PrivateAttr(default_factory=dict)  # mutable-ok: call metadata


VideoCancelResult: TypeAlias = VideoCancelObject | VideoCancelRefusal


@dataclass(frozen=True, slots=True)
class VideoCancelRequest:
    """Where a provider's task status is read and how its cancel is sent."""

    status_url: str
    cancel_method: Literal["PUT", "DELETE", "POST"]
    cancel_url: str
    recheck_url: str | None = None


@dataclass(frozen=True, slots=True)
class VideoCancelAccepted:
    outcome: VideoCancelOutcome
    provider_status: str
    progress: float | None = None


@dataclass(frozen=True, slots=True)
class VideoCancelProceed:
    """The status read allows a cancel; ``if_accepted`` is the result when the provider takes it."""

    if_accepted: VideoCancelAccepted


VideoCancelVerdict: TypeAlias = VideoCancelAccepted | VideoCancelRefusal
VideoCancelPreflight: TypeAlias = VideoCancelProceed | VideoCancelVerdict
