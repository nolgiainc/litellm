import base64
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

import httpx
from httpx._types import RequestFiles
from typing_extensions import ReadOnly

import litellm
from litellm.constants import DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.litellm_core_utils.url_utils import safe_get
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.interactions import InteractionsAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from .transformation import fetch_image_as_base64

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.videos.capabilities import CapabilityParamSupport as _CapabilityParamSupport

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
    CapabilityParamSupport = _CapabilityParamSupport
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any
    CapabilityParamSupport = Any

INTERACTIONS_API_REVISION = "2026-05-20"

_OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO: Mapping[str, str] = {
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
}

_SUPPORTED_ASPECT_RATIOS = frozenset({"16:9", "9:16"})

_TERMINAL_FAILURE_STATUSES = frozenset({"failed", "cancelled", "incomplete", "budget_exceeded"})


def _map_interaction_status(status: str | None) -> str:
    if status == "completed":
        return "completed"
    if status in _TERMINAL_FAILURE_STATUSES:
        return "failed"
    return "processing"


def _find_video_part(interaction: InteractionsAPIResponse) -> dict[str, Any] | None:
    steps = interaction.steps or interaction.outputs or []
    for step in reversed(steps):
        if step.get("type") != "model_output":
            continue
        for part in step.get("content") or []:
            if part.get("type") == "video":
                return part
    return None


_CAPABILITY_PARAMS = frozenset(
    (
        "input_reference",
        "image_url",
        "negative_prompt",
        "video_urls",
    )
)

# Omni EDIT mode (https://ai.google.dev/gemini-api/docs/omni, "Edit your own
# videos"): the customer's clip is the source the interaction rewrites, and the
# prompt names only the change ("Add fog. Keep everything else the same."). The
# source rides the fal-shaped `video_urls` slot so nolgia-api's reference-video
# plumbing (video_asset_ids -> signed URL) reaches this provider unchanged; Omni
# edits exactly ONE source clip, so a second entry is refused rather than dropped.
_MAX_SOURCE_VIDEOS: Final = 1

# The source clip always rides INLINE (base64 `data` part). Google caps an inline
# request at 20 MB, so the raw clip is capped where its base64 form plus the rest
# of the body still fits: 14 MiB of bytes is 19.57 MB encoded, leaving ~0.4 MB for
# the prompt and JSON. The Files API route Google suggests above that is NOT
# used: its upload and ACTIVE polling are synchronous, and this transform also
# runs on the async handler's event loop, so a large source would stall the
# proxy worker for the whole upload. A larger source is refused with a message
# that says why; nolgia-api caps the same size up front.
_SOURCE_VIDEO_MAX_BYTES: Final = 14 * 1024 * 1024

_DEFAULT_VIDEO_MIME_TYPE: Final = "video/mp4"

# Interactions VideoContent.mime_type enum (https://ai.google.dev/api/interactions-api).
_SUPPORTED_VIDEO_MIME_TYPES: Final = frozenset(
    (
        "video/mp4",
        "video/mpeg",
        "video/mpg",
        "video/mov",
        "video/quicktime",
        "video/avi",
        "video/x-flv",
        "video/webm",
        "video/wmv",
        "video/x-ms-wmv",
        "video/3gpp",
    )
)

# A signed object URL can omit Content-Type or answer application/octet-stream;
# the path extension and then the container's magic bytes decide instead of
# silently declaring every unknown body as MP4.
_VIDEO_MIME_BY_EXTENSION: Final = MappingProxyType(
    {
        ".mp4": "video/mp4",
        ".m4v": "video/mp4",
        ".mov": "video/mov",
        ".qt": "video/mov",
        ".webm": "video/webm",
        ".mpeg": "video/mpeg",
        ".mpg": "video/mpg",
        ".avi": "video/avi",
        ".flv": "video/x-flv",
        ".wmv": "video/wmv",
        ".3gp": "video/3gpp",
    }
)


class _VideoPart(TypedDict):
    type: ReadOnly[Literal["video"]]
    mime_type: ReadOnly[str]
    data: ReadOnly[str]


class _TextPart(TypedDict):
    type: ReadOnly[Literal["text"]]
    text: ReadOnly[str]


class _EditVideoConfig(TypedDict):
    task: ReadOnly[Literal["edit"]]


class _EditGenerationConfig(TypedDict):
    video_config: ReadOnly[_EditVideoConfig]


def _sniffed_video_mime_type(content: bytes) -> str | None:
    """Container magic bytes: ISO BMFF (`ftyp` at offset 4, QuickTime brand `qt  `), Matroska/WebM, AVI, FLV."""
    if content[4:8] == b"ftyp":
        return "video/mov" if content[8:12] == b"qt  " else _DEFAULT_VIDEO_MIME_TYPE
    if content[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    if content[:4] == b"RIFF" and content[8:12] == b"AVI ":
        return "video/avi"
    if content[:3] == b"FLV":
        return "video/x-flv"
    return None


def _video_mime_type(response: httpx.Response, video_url: str, content: bytes) -> str:
    content_type: Final = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if content_type == "video/quicktime":
        # The Interactions enum spells QuickTime as video/mov.
        return "video/mov"
    if content_type in _SUPPORTED_VIDEO_MIME_TYPES:
        return content_type
    path: Final = httpx.URL(video_url).path.lower()
    extension: Final = path[path.rfind(".") :] if "." in path.rsplit("/", 1)[-1] else ""
    by_extension: Final = _VIDEO_MIME_BY_EXTENSION.get(extension)
    if by_extension is not None:
        return by_extension
    sniffed: Final = _sniffed_video_mime_type(content)
    if sniffed is not None:
        return sniffed
    raise ValueError(
        f"Omni edit source video has an unrecognized type (Content-Type {content_type or 'missing'!r}, no known "
        "extension or container signature); send an MP4, MOV, WebM, MPEG, AVI, FLV, WMV or 3GP clip"
    )


def _source_video_part(video_url: str) -> _VideoPart:
    """
    Encode the customer's source clip as the Omni video input part for EDIT mode.

    The URL is caller controlled (a signed asset URL), so it is fetched through the
    SSRF checked helper exactly like the start frame, with a Range header asking
    for at most one byte past the cap: a server that honors Range (GCS signed URLs
    and every mainstream CDN do) never sends more than that, so an oversized
    source costs a bounded read instead of worker memory. A body over the cap,
    whether a 206 at the boundary or a server that ignored Range, is refused.
    """
    response: Final = safe_get(
        litellm.module_level_client,
        video_url,
        headers={"Range": f"bytes=0-{_SOURCE_VIDEO_MAX_BYTES}"},  # mutable-ok: httpx takes a concrete headers dict
    )
    response.raise_for_status()
    content: Final = response.content
    if not content:
        raise ValueError("Omni edit source video downloaded as zero bytes")
    if len(content) > _SOURCE_VIDEO_MAX_BYTES:
        raise ValueError(
            f"Omni edit source video is larger than {_SOURCE_VIDEO_MAX_BYTES // (1024 * 1024)} MB, the inline limit "
            "(Google caps an inline request at 20 MB); re-encode or trim the clip"
        )
    part: Final[_VideoPart] = {
        "type": "video",
        "mime_type": _video_mime_type(response, video_url, content),
        "data": base64.b64encode(content).decode("utf-8"),
    }
    return part


def _source_video_urls(value: object) -> tuple[str, ...]:
    """video_urls arrives as a list of URL strings (fal shape); a lone string is tolerated."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        items: Final[tuple[object, ...]] = tuple(value)  # pyright: ignore[reportUnknownArgumentType]  # OpenAI-shaped video params are untyped at this boundary
        urls: Final = tuple(item for item in items if isinstance(item, str) and item)
        if len(urls) != len(items):
            raise ValueError("video_urls must be a list of https URL strings")
        return urls
    raise ValueError("video_urls must be a list of https URL strings")


def _requested_seconds(logging_obj: "LiteLLMLoggingObj") -> float:
    """
    The billed length for the cost log: the request's `seconds` when the caller sent
    one, else Google's 8 s default. Omni takes no duration field (generation gets it
    as a prompt clause, an edit follows its source), so the Interactions response
    never states it; the caller's own value is the only exact basis. nolgia-api
    sends an edit's source length, rounded up, as `seconds` for exactly this.
    """
    optional_params: Final[object] = logging_obj.model_call_details.get("optional_params")
    seconds: Final[object] = optional_params.get("seconds") if isinstance(optional_params, dict) else None
    if isinstance(seconds, (int, float, str)):
        try:
            parsed: Final = float(seconds)
        except ValueError:
            return float(DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS)
        if parsed > 0:
            return parsed
    return float(DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS)


def _start_frame_part(start_frame: FileTypes) -> dict[str, str]:
    """
    Encode the start frame as an Omni image input part.

    input_reference and image_url are the same slot under two names and both are
    declared as executable, so both are honored here. A hosted URL is fetched (via
    the SSRF-checked helper), while a multipart /v1/videos upload arrives as bytes or
    a file-like object and is inlined directly; ignoring either would send the
    interaction as text-to-video and bill for a result without the requested frame.
    """
    if isinstance(start_frame, str):
        base64_data, mime_type = fetch_image_as_base64(start_frame)
        # mutable-ok: request part dict, handed straight to the JSON body
        return {"type": "image", "data": base64_data, "mime_type": mime_type}
    extracted = extract_file_data(start_frame)
    content_type = extracted.get("content_type") or ""
    if not content_type or content_type == "application/octet-stream":
        content_type = "image/png"
    return {  # mutable-ok: request part dict, handed straight to the JSON body
        "type": "image",
        "data": base64.b64encode(extracted["content"]).decode("utf-8"),
        "mime_type": content_type,
    }


class GeminiOmniVideoConfig(BaseVideoConfig):
    """
    Video generation for Gemini Omni models (e.g. gemini-omni-flash-preview).

    Unlike Veo, Omni models generate video through the Interactions API:
    1. POST /v1beta/interactions with background=true returns an interaction id
    2. Poll GET /v1beta/interactions/{id} until status is terminal
    3. The completed interaction carries the video inline as base64 in the
       model_output step (or as a files URI for large outputs)

    Omni has no explicit duration/negative-prompt parameters; per Google's
    prompt guide both are expressed in the prompt text, which is what
    transform_video_create_request does with ``seconds`` and ``negative_prompt``.
    """

    def get_capability_param_support(self, model: str) -> "CapabilityParamSupport":
        """
        Omni executes a start frame: transform_video_create_request reads image_url
        and switches the interaction to image_to_video. It also executes
        negative_prompt, though not as a field; Omni has no negative channel, so the
        same method folds it into the prompt as an explicit exclusion, which is a
        real constraint on the render rather than a discarded param. It executes ONE
        reference video as the EDIT source (video_urls[0]): the interaction rewrites
        that clip under the prompt's instruction with task=edit, which is the lane
        that adds VFX to, or re-angles, footage the customer shot. It has no
        end-frame, image-element, audio-reference, regeneration or bitrate surface,
        and its audio is native with no generate_audio switch.
        """
        from litellm.videos.capabilities import DeclaredCapabilityParams

        return DeclaredCapabilityParams(_CAPABILITY_PARAMS)

    def get_supported_openai_params(self, model: str) -> list:
        return ["model", "prompt", "seconds", "size"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, Any]:
        mapped_params: dict[str, Any] = {}

        size = video_create_optional_params.get("size")
        if size:
            aspect_ratio = _OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO.get(size)
            if aspect_ratio:
                mapped_params["aspect_ratio"] = aspect_ratio

        for key, value in video_create_optional_params.items():
            if key not in {"model", "prompt", "size"} and key not in mapped_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("GOOGLE_API_KEY") or get_secret_str("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini Omni video generation. "
                "Set it via environment variable or pass it as api_key parameter."
            )

        headers.update(
            {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
                "Api-Revision": INTERACTIONS_API_REVISION,
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"

        if not model or model == "":
            return api_base.rstrip("/")

        return f"{api_base.rstrip('/')}/v1beta/interactions"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        params = video_create_optional_request_params

        seconds = params.get("seconds") or params.get("duration_seconds")
        negative_prompt = params.get("negative_prompt")
        aspect_ratio = params.get("aspect_ratio")
        start_frame = params.get("image_url") or params.get("input_reference")
        source_videos: Final = _source_video_urls(params.get("video_urls"))

        if source_videos and start_frame:
            # An explicit edit task disables multimodal reference inputs (Google's
            # cookbook), and a start frame has no meaning when the source clip
            # supplies every frame; refusing beats silently dropping either.
            raise ValueError(
                "Gemini Omni edit mode takes the source clip only: send video_urls without image_url / input_reference"
            )
        if len(source_videos) > _MAX_SOURCE_VIDEOS:
            raise ValueError(
                f"Gemini Omni edits exactly one source video per request; got {len(source_videos)} video_urls"
            )

        prompt_parts: list[str] = [prompt]
        # Edit output follows the source clip, so a duration clause would fight the
        # source; it is folded in for generation only.
        if seconds and not source_videos:
            prompt_parts.append(f"The video must be exactly {seconds} seconds long.")
        if negative_prompt:
            prompt_parts.append(f"Do not include: {negative_prompt}.")
        full_prompt = " ".join(prompt_parts)

        response_format: dict[str, Any] = {"type": "video"}
        # The edit inherits the source clip's framing; an aspect ratio only applies
        # to generation.
        if aspect_ratio in _SUPPORTED_ASPECT_RATIOS and not source_videos:
            response_format["aspect_ratio"] = aspect_ratio

        request_data: dict[str, Any] = {
            "model": model.replace("gemini/", ""),
            "input": full_prompt,
            "response_format": response_format,
            "background": True,
            "store": True,
        }

        if source_videos:
            text_part: Final[_TextPart] = {"type": "text", "text": full_prompt}
            edit_input: Final = [  # mutable-ok: Interactions requires a JSON array of ordered input parts
                _source_video_part(source_videos[0]),
                text_part,
            ]
            edit_video_config: Final[_EditVideoConfig] = {"task": "edit"}
            edit_generation_config: Final[_EditGenerationConfig] = {"video_config": edit_video_config}
            request_data["input"] = edit_input
            request_data["generation_config"] = edit_generation_config
        elif start_frame:
            request_data["input"] = [
                _start_frame_part(start_frame),
                {"type": "text", "text": full_prompt},
            ]
            request_data["generation_config"] = {"video_config": {"task": "image_to_video"}}

        return request_data, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        try:
            raw_json = raw_response.json()
        except ValueError as e:
            raise ValueError(f"Failed to parse interaction response: {e}")
        interaction = InteractionsAPIResponse(**raw_json)

        interaction_id = interaction.id
        if not interaction_id:
            raise ValueError(f"No interaction id in Gemini Omni response: {raw_response.text}")

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(interaction_id, custom_llm_provider, model)
        else:
            video_id = interaction_id

        video_obj = VideoObject(
            id=video_id,
            object="video",
            status=_map_interaction_status(interaction.status),
            model=model,
        )
        video_obj.usage = {
            "duration_seconds": _requested_seconds(logging_obj),
            "video_resolution": "720p",
        }
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        interaction_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/v1beta/interactions/{interaction_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response_json = raw_response.json()
        interaction = InteractionsAPIResponse(**response_json)

        interaction_id = interaction.id or ""
        if custom_llm_provider:
            video_id = encode_video_id_with_provider(interaction_id, custom_llm_provider, None)
        else:
            video_id = interaction_id

        status = _map_interaction_status(interaction.status)

        error_data: dict[str, Any] | None = None
        if status == "failed":
            error_data = response_json.get("error") or {
                "code": "interaction_" + (interaction.status or "failed"),
                "message": f"Gemini Omni video generation ended with status {interaction.status!r}.",
            }

        return VideoObject(
            id=video_id,
            object="video",
            status=status,
            error=error_data,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        interaction_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/v1beta/interactions/{interaction_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        interaction = InteractionsAPIResponse(**raw_response.json())

        status = _map_interaction_status(interaction.status)
        if status == "processing":
            raise ValueError(
                "Video generation is not complete yet. Please check status with video_status() before downloading."
            )
        if status == "failed":
            raise ValueError(f"Gemini Omni video generation ended with status {interaction.status!r}.")

        video_part = _find_video_part(interaction)
        if video_part is None:
            raise ValueError("No video output in completed interaction.")

        inline_data = video_part.get("data")
        if inline_data:
            return base64.b64decode(inline_data)

        uri = video_part.get("uri")
        if uri:
            download_headers: dict[str, str] = {}
            api_key = raw_response.request.headers.get("x-goog-api-key")
            if api_key:
                download_headers["x-goog-api-key"] = api_key
            download_response = litellm.module_level_client.get(url=uri, headers=download_headers)
            download_response.raise_for_status()
            return download_response.content

        raise ValueError("Video output has neither inline data nor a download URI.")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported for Gemini Omni via the videos API.")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported for Gemini Omni via the videos API.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video list is not supported for Gemini Omni.")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video list is not supported for Gemini Omni.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video delete is not supported for Gemini Omni.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported for Gemini Omni.")

    def transform_video_create_character_request(self, name, video, api_base, litellm_params, headers):
        raise NotImplementedError("video create character is not supported for Gemini Omni")

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video create character is not supported for Gemini Omni")

    def transform_video_get_character_request(self, character_id, api_base, litellm_params, headers):
        raise NotImplementedError("video get character is not supported for Gemini Omni")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for Gemini Omni")

    def transform_video_edit_request(
        self,
        prompt,
        video_id,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
        prefetched_source_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini Omni via the videos API")

    def transform_video_edit_response(
        self,
        raw_response,
        logging_obj,
        custom_llm_provider=None,
        request_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini Omni via the videos API")

    def transform_video_extension_request(
        self,
        prompt,
        video_id,
        seconds,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
    ):
        raise NotImplementedError("video extension is not supported for Gemini Omni")

    def transform_video_extension_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("video extension is not supported for Gemini Omni")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ..common_utils import GeminiError

        return GeminiError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
