"""
Kling's PATH-BASED video surface: Kling 3.0 Turbo, Kling 3.0 Omni, Kling O1.

These three models do not exist on the classic `/v1` API at all - it answers
`kling-v3-turbo` and `kling-o1` with `1203 not supported for this API` and
`kling-v3-omni` with `1201 model is not supported` (measured NOL-994). They
live only on a newer surface that differs from the classic one in every way
that matters to a transform, which is why this is a separate config rather
than a branch inside KlingVideoConfig:

  * the model is the PATH (`POST /text-to-video/kling-3.0-turbo`) instead of a
    `model_name` field in the body;
  * knobs live under `settings.*` (resolution/duration/aspect_ratio/audio)
    instead of at the top level, and resolution is the public label
    (`720p`/`1080p`) rather than the classic `mode` (`std`/`pro`);
  * media and prompt arrive as a typed `contents[]` array rather than as
    `prompt` + `image`;
  * auth is a plain bearer of a single opaque CONSOLE key, not an HS256 JWT
    signed with an AccessKey:SecretKey pair. The classic credential is refused
    outright here (`401 code 1002`), so the two surfaces need two secrets;
  * polling is one shared `GET /tasks?task_ids=...` whose `data` is a LIST,
    not a per-kind `GET /v1/videos/{kind}/{id}` whose `data` is an object.

Everything measured against the live surface on 2026-09-19 (NOL-1041), by the
zero-cost technique from NOL-994 - keep exactly one field invalid so nothing
can submit. Findings that shaped this file:

  * Validation order is `settings.aspect_ratio` -> `settings.resolution` ->
    `settings.audio` -> prompt/contents-text blankness -> `settings.duration`
    on the prompt-shaped endpoints, and duration BEFORE the contents-text
    check on the contents-shaped ones. An empty prompt-content item is
    therefore the universal zero-cost guard on `/omni-video/*`.
  * TRULY unknown fields are SILENTLY IGNORED (`{"zzz_not_a_field": true}`
    changes nothing). That is the motion-control behaviour, not the classic
    `sound` behaviour, so no capability may be declared here on the strength
    of the docs alone.
  * BUT `settings.audio` is a REAL validated field on EVERY path endpoint,
    including `/text-to-video/kling-3.0-turbo`, where Kling does not document
    it: `"bogus"` is rejected by name while `native`/`off` pass. And it is
    honoured, not just accepted - a 5 s Omni render at `audio: off` came back
    video-only and drew 3.0 units (0.6 U/s), the same render at
    `audio: native` came back h264+aac and drew 4.0 units (0.8 U/s), which are
    Kling's two published Omni rates to the unit.

Because the video cost path can only tier a rate by RESOLUTION, an audio
toggle at one resolution cannot be priced on a single route. So audio is NOT a
per-request knob here: it is baked into the model id, one id per published
rate row, and the transform always sends the value that id was priced for.
`kling-3.0-omni` is the silent rate, `kling-3.0-omni-audio` the native-audio
rate, and 3.0 Turbo has no silent rate published at all so it is always native.

Kling's own meter agreed with the published table to the unit on all four
measured renders, so every rate below is measured rather than inferred.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.kling.auth import kling_console_auth_headers
from litellm.llms.kling.common_utils import (
    KLING_TASK_STATUS_MAP,
    resolve_kling_path_api_base,
    strip_kling_prefix,
)
from litellm.llms.kling.videos.transformation import KlingVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)
from litellm.videos.capabilities import CapabilityParamSupport, DeclaredCapabilityParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj  # rebind-ok: the type-checking half of a TYPE_CHECKING alias
else:
    # Imported lazily by the logging module, so the runtime alias is the widest
    # non-Any stand-in rather than the real class.
    LiteLLMLoggingObj = object  # rebind-ok: the runtime half of a TYPE_CHECKING alias


# The `model_id` slot of an encoded video id. The path surface polls one shared
# /tasks endpoint, so unlike the classic surface there is no per-kind path to
# remember - this value exists only to route a later status or content lookup
# back to THIS config, since the retrieve path resolves a provider config from
# the decoded model_id and not from the original model name.
PATH_TASK_KIND: Final = "kling-path-tasks"

_SIZE_TO_ASPECT_RATIO: Final[Mapping[str, str]] = MappingProxyType(
    {  # mutable-ok: frozen lookup table
        "1280x720": "16:9",
        "1920x1080": "16:9",
        "720x1280": "9:16",
        "1080x1920": "9:16",
        "1024x1024": "1:1",
        "1080x1080": "1:1",
    }
)

# Vendor knobs a caller may pass through untouched. The first three ride in
# `options`; multi_shot is a `settings` field on the Omni endpoints.
_VENDOR_OPTIONS: Final = ("external_task_id", "callback_url", "watermark_info")
_PASSTHROUGH_OPTIONS: Final = (*_VENDOR_OPTIONS, "multi_shot")

_DEFAULT_RESOLUTION: Final = "720p"
_DEFAULT_DURATION: Final = 5


@dataclass(frozen=True, slots=True)
class _PathModel:
    """One published Kling rate row, as an endpoint plus the settings it allows."""

    t2v_path: str
    i2v_path: str
    # True when the text-to-video form also takes the typed contents[] array.
    # Only 3.0 Turbo's /text-to-video endpoint takes a flat `prompt` string;
    # every other path endpoint here is contents-shaped.
    t2v_uses_contents: bool
    resolutions: frozenset[str]
    durations: frozenset[int]
    # The settings.audio value this id is priced for, always sent verbatim.
    # None means the endpoint publishes no silent rate and takes no audio
    # field, i.e. the render is natively audible and billed that way.
    audio: str | None
    # Whether the output carries an audio track, for capability advertisement.
    audible: bool


_OMNI_DURATIONS: Final = frozenset(range(3, 16))
_O1_DURATIONS: Final = frozenset(range(3, 11))
_HD_TIERS: Final = frozenset(("720p", "1080p"))

PATH_MODELS: Final[Mapping[str, _PathModel]] = MappingProxyType(
    {  # mutable-ok: frozen registry of path-based Kling routes
        # 0.8 U/s at 720p, 1.0 U/s at 1080p, native audio, no silent row
        # published and no 4K row at all ("video resolution value '4k' is
        # invalid" on this endpoint). Measured: 5 s at 720p drew 4.0 units and
        # came back h264+aac.
        "kling-3.0-turbo": _PathModel(
            t2v_path="/text-to-video/kling-3.0-turbo",
            i2v_path="/image-to-video/kling-3.0-turbo",
            t2v_uses_contents=False,
            resolutions=_HD_TIERS,
            durations=_OMNI_DURATIONS,
            audio=None,
            audible=True,
        ),
        # 0.6 U/s at 720p, 0.8 at 1080p. Measured: 3.0 units for 5 s, silent.
        "kling-3.0-omni": _PathModel(
            t2v_path="/omni-video/kling-3.0-omni",
            i2v_path="/omni-video/kling-3.0-omni",
            t2v_uses_contents=True,
            resolutions=_HD_TIERS,
            durations=_OMNI_DURATIONS,
            audio="off",
            audible=False,
        ),
        # 0.8 U/s at 720p, 1.0 at 1080p. Measured: 4.0 units for 5 s, h264+aac.
        "kling-3.0-omni-audio": _PathModel(
            t2v_path="/omni-video/kling-3.0-omni",
            i2v_path="/omni-video/kling-3.0-omni",
            t2v_uses_contents=True,
            resolutions=_HD_TIERS,
            durations=_OMNI_DURATIONS,
            audio="native",
            audible=True,
        ),
        # 0.6 U/s at 720p, 0.8 at 1080p. Measured: 3.0 units for 5 s, silent.
        # O1's only audio value is `original`, which carries the audio of an
        # INPUT VIDEO - and a video input moves the row to 0.9/1.2 U/s, a
        # different rate this route is not priced for, so neither is exposed.
        # Duration tops out at 10 here, not 15: `11` is rejected, `10` passes.
        "kling-o1": _PathModel(
            t2v_path="/omni-video/kling-o1",
            i2v_path="/omni-video/kling-o1",
            t2v_uses_contents=True,
            resolutions=_HD_TIERS,
            durations=_O1_DURATIONS,
            audio="off",
            audible=False,
        ),
    }
)


def _json_mapping(raw_response: httpx.Response) -> Mapping[str, object]:
    """A JSON body as a typed mapping; a non-object body is an empty one."""
    parsed: Final[object] = raw_response.json()  # pyright: ignore[reportAny]  # httpx types .json() as Any; narrowed by _as_mapping
    return _as_mapping(parsed)


def _as_mapping(value: object) -> Mapping[str, object]:
    """A value as a read-only mapping; anything else reads as an empty one."""
    return value if isinstance(value, Mapping) else {}  # mutable-ok: empty payload for a non-object value


def _billed_seconds(duration: object) -> float | None:
    """The submitted duration as a number, or None when it cannot be one."""
    try:
        return float(duration)  # pyright: ignore[reportArgumentType]  # guarded by the except below
    except (TypeError, ValueError):
        return None


def is_kling_path_model(model: str | None) -> bool:
    """
    True for a model this config serves, or for a video id minted by it.

    Both forms have to answer here because a status or content lookup resolves
    its provider config from the `model_id` encoded into the video id rather
    than from the model name the create used.
    """
    if not model:
        return False
    if model == PATH_TASK_KIND:
        return True
    try:
        return strip_kling_prefix(model) in PATH_MODELS
    except ValueError:
        return False


class KlingPathVideoConfig(KlingVideoConfig):
    """
    Kling 3.0 Turbo / 3.0 Omni / O1 on the path-based API.

    Subclasses the classic config only for the pieces that are genuinely
    shared - the body-code error mapping, the rate-limit handling and the
    `NotImplementedError`s for remix/list/delete. Every request and response
    transform is overridden; nothing about the classic `mode`/`model_name`
    shape survives.
    """

    def _model(self, model: str) -> _PathModel:
        bare: Final = strip_kling_prefix(model)
        spec: Final = PATH_MODELS.get(bare)
        if spec is None:
            raise litellm.BadRequestError(
                message=f"'{model}' is not a path-based Kling model; expected one of {sorted(PATH_MODELS)}.",
                model=model,
                llm_provider=litellm.LlmProviders.KLING.value,
            )
        return spec

    def supports_promptless_video_create(self, model: str) -> bool:
        return False

    def get_capability_param_support(self, model: str) -> CapabilityParamSupport:
        """
        A start frame, and nothing else.

        `generate_audio` is deliberately NOT declared even though every one of
        these endpoints validates `settings.audio`: audio is priced at a
        different rate at the SAME resolution, and the video cost path tiers
        only by resolution, so a per-request toggle would bill one rate for the
        other. The choice is made by model id instead - see the module
        docstring - which is the same reasoning the silent 2.6/2.5 rows carry.

        No end frame: `last_frame` exists on the Omni endpoints, but Kling
        prices it with the rest of the row and it is untested here; declaring
        an undeclared-but-silently-ignored field is exactly the failure this
        gate exists to prevent, since this surface DOES silently drop unknown
        fields.
        """
        return DeclaredCapabilityParams(frozenset(("input_reference", "image_url")))

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig's signature
        return [  # mutable-ok: BaseVideoConfig requires a list of supported parameter names
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig's signature
        supplied: Final[Mapping[str, object]] = dict(  # mutable-ok: merged once, then read only
            video_create_optional_params
        )
        extra_body: Final = supplied.get("extra_body")
        params: Final[Mapping[str, object]] = (
            {**supplied, **extra_body}  # mutable-ok: merged once, then read only
            if isinstance(extra_body, dict)
            else supplied
        )

        spec: Final = self._model(model)
        reference: Final = params.get("input_reference") or params.get("image_url")
        start_image: Final = self._coerce_start_image(reference)  # pyright: ignore[reportArgumentType]  # FileTypes is not expressible from a loosely typed params mapping

        if params.get("generate_audio") is not None:
            raise litellm.BadRequestError(
                message=(
                    f"Kling model '{model}' takes no per-request audio flag: audio changes the per-second rate at "
                    "the same resolution, and the cost path tiers only by resolution. Use the "
                    f"'{'kling-3.0-omni-audio' if spec.audio == 'off' else 'kling-3.0-omni'}' id, which is priced "
                    "for the audio setting it always sends."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.KLING.value,
            )

        return {  # mutable-ok: the video pipeline mutates the mapped optional-parameter dict
            key: value
            for key, value in (
                ("resolution", self._validated_resolution(params.get("resolution"), spec, model)),
                ("duration", self._validated_duration(params.get("seconds"), spec, model)),
                ("aspect_ratio", self._aspect_ratio(params.get("size"))),
                ("image", start_image),
                *((key, params.get(key)) for key in _PASSTHROUGH_OPTIONS),
            )
            if value is not None
        }

    @staticmethod
    def _aspect_ratio(size: object) -> str | None:
        if not isinstance(size, str):
            return None
        return _SIZE_TO_ASPECT_RATIO.get(size) or (size.replace("x", ":") if "x" in size else None)

    @staticmethod
    def _validated_resolution(resolution: object, spec: _PathModel, model: str) -> str:
        label: Final = str(resolution).strip().lower() if resolution is not None else _DEFAULT_RESOLUTION
        if label not in spec.resolutions:
            raise litellm.BadRequestError(
                message=(
                    f"Kling model '{model}' does not publish a '{label}' rate; use one of "
                    f"{sorted(spec.resolutions)}. Rendering at an unpriced tier would record no COGS."
                ),
                model=model,
                llm_provider=litellm.LlmProviders.KLING.value,
            )
        return label

    @staticmethod
    def _validated_duration(seconds: object, spec: _PathModel, model: str) -> int:
        if seconds is None:
            return _DEFAULT_DURATION
        bounds: Final = f"{min(spec.durations)} through {max(spec.durations)}"
        error: Final = litellm.BadRequestError(
            message=f"Kling model '{model}' takes an integer seconds value from {bounds}.",
            model=model,
            llm_provider=litellm.LlmProviders.KLING.value,
        )
        try:
            value: Final = float(seconds)  # pyright: ignore[reportArgumentType]  # guarded by the except below
        except (TypeError, ValueError) as exc:
            raise error from exc
        if (
            isinstance(seconds, bool)
            or not math.isfinite(value)
            or int(value) != value
            or int(value) not in spec.durations
        ):
            raise error
        return int(value)

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseVideoConfig's signature
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig's signature
        resolved: Final = api_key or (litellm_params.api_key if litellm_params else None)
        return {  # mutable-ok: validate_environment returns a mutable header dict
            **headers,
            **kling_console_auth_headers(resolved),
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseVideoConfig's signature
    ) -> str:
        return resolve_kling_path_api_base(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,  # mutable-ok: BaseVideoConfig's signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig's signature
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig's signature
        spec: Final = self._model(model)
        mapped: Final[Mapping[str, object]] = {  # mutable-ok: read-only copy of the mapped params
            key: value for key, value in video_create_optional_request_params.items() if key != "model"
        }
        text: Final = (prompt or "").strip()
        image: Final = mapped.get("image")
        if not text and not image:
            raise litellm.BadRequestError(
                message=f"Kling model '{model}' requires a prompt, a start image, or both.",
                model=model,
                llm_provider=litellm.LlmProviders.KLING.value,
            )

        settings: Final = {  # mutable-ok: nested vendor JSON body, built once and serialised
            key: value
            for key, value in (
                ("resolution", mapped.get("resolution", _DEFAULT_RESOLUTION)),
                ("duration", mapped.get("duration", _DEFAULT_DURATION)),
                ("aspect_ratio", mapped.get("aspect_ratio")),
                ("multi_shot", mapped.get("multi_shot")),
                # Always sent, never taken from the caller: this id is priced for it.
                ("audio", spec.audio),
            )
            if value is not None
        }
        options: Final = {  # mutable-ok: nested vendor JSON body, built once and serialised
            key: mapped[key] for key in _VENDOR_OPTIONS if mapped.get(key) is not None
        }
        contents: Final = [  # mutable-ok: nested vendor JSON body, built once and serialised
            item
            for item in (
                {"type": "prompt", "text": text} if text else None,  # mutable-ok: one JSON content item
                {"type": "first_frame", "url": image} if image else None,  # mutable-ok: one JSON content item
            )
            if item is not None
        ]

        body: Final = {  # mutable-ok: the HTTP video handler requires a JSON-serializable dict
            "settings": settings,
            **({"options": options} if options else {}),  # mutable-ok: empty merge operand
            **(
                {"contents": contents}  # mutable-ok: vendor JSON body, built once and serialised
                if (image or spec.t2v_uses_contents)
                else {"prompt": text}  # mutable-ok: vendor JSON body, built once and serialised
            ),
        }
        files: Final[RequestFiles] = []  # mutable-ok: this surface takes JSON only, never multipart
        return body, files, f"{api_base}{spec.i2v_path if image else spec.t2v_path}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,  # mutable-ok: BaseVideoConfig's signature
    ) -> VideoObject:
        response_data: Final[Mapping[str, object]] = _json_mapping(raw_response)
        # _raise_for_kling_error reads the classic dict shape; the copy is the adapter.
        self._raise_for_kling_error(dict(response_data))  # mutable-ok: one-shot copy for the shared error mapper

        data: Final = _as_mapping(response_data.get("data"))
        task_id: Final = data.get("id")
        if not task_id or not isinstance(task_id, str):
            raise ValueError(f"Kling video submit response is missing a string data.id: {response_data}")

        settings: Final = _as_mapping(_as_mapping(request_data).get("settings"))
        # Kling bills the REQUESTED duration, not the delivered one: every
        # measured 5 s render returned a 5.041 s file and drew exactly 5x the
        # published per-second rate. So the submitted value is the billing
        # basis, and taking it from the request is exact rather than an
        # approximation of the output.
        billed: Final = _billed_seconds(settings.get("duration"))
        seconds: Final = str(billed) if billed is not None else None
        usage: Final[dict[str, float | str]] = {  # mutable-ok: VideoObject.usage is a plain dict
            key: value
            for key, value in (
                ("duration_seconds", billed),
                # The tier key the per-second price map is indexed by. Unlike
                # the classic surface there is no mode to invert - the request
                # already carries the public label.
                (
                    "video_resolution",
                    str(settings["resolution"]) if settings.get("resolution") is not None else None,
                ),
            )
            if value is not None
        }

        video_obj: Final = VideoObject(
            id=task_id,
            object="video",
            status=KLING_TASK_STATUS_MAP.get(str(data.get("status", "submitted")), "queued"),
            model=model,
            seconds=seconds,
            size=str(settings["aspect_ratio"]).replace(":", "x") if settings.get("aspect_ratio") else None,
            usage=usage,
        )
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, PATH_TASK_KIND)
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig's signature
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig's signature
        return self._build_task_url(video_id, api_base), {}  # mutable-ok: empty literal, never mutated

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        try:
            response_data: Final[Mapping[str, object]] = _json_mapping(raw_response)
        except (ValueError, JSONDecodeError):
            return VideoObject(id="", object="video", status="in_progress")

        task: Final = self._first_task(response_data)
        task_id: Final = str(task.get("id") or "")
        status: Final = KLING_TASK_STATUS_MAP.get(str(task.get("status", "submitted")), "queued")

        message: Final = task.get("message") or response_data.get("message") or "Video generation failed"
        error: Final = (
            {"code": "failed", "message": str(message)}  # mutable-ok: VideoObject.error is a plain dict
            if status == "failed"
            else None
        )

        video_obj: Final = VideoObject(id=task_id, object="video", status=status, error=error)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, PATH_TASK_KIND)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig's signature
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig's signature
        return self._build_task_url(video_id, api_base), {}  # mutable-ok: empty literal, never mutated

    def _build_task_url(self, video_id: str, api_base: str) -> str:
        decoded: Final = decode_video_id_with_provider(video_id)
        task_id: Final = decoded.get("video_id") or extract_original_video_id(video_id)
        if not task_id:
            raise ValueError("Kling path-based status/content lookup requires the id returned by video creation.")
        # A task id is Kling's own decimal snowflake, so it is validated as one
        # rather than escaped: it becomes a QUERY value here, and anything that
        # is not digits did not come from a create response.
        if not str(task_id).isdigit():
            raise ValueError(f"Kling task id must be numeric; got '{task_id}'.")
        return f"{api_base}/tasks?task_ids={task_id}"

    @staticmethod
    def _first_task(response_data: Mapping[str, object]) -> Mapping[str, object]:
        """
        `GET /tasks` answers with a LIST under `data`, even for one id.

        An empty list is a genuinely unknown task rather than a transport
        failure, so it degrades to an empty mapping and the caller reports
        `queued` - the same shape an unparseable body takes.
        """
        data: Final = response_data.get("data")
        if isinstance(data, Sequence) and not isinstance(data, (str, bytes)) and data:
            return _as_mapping(data[0])
        return _as_mapping(None)

    @staticmethod
    def _extract_video_url(response_data: Mapping[str, object]) -> str:
        """
        The path surface's result shape, replacing the classic
        `data.task_result.videos[0].url`.

        Overriding this - rather than the two content transforms that call it -
        keeps the inherited sync and async download paths, so there is one
        place that knows where a finished clip lives.
        """
        task: Final = KlingPathVideoConfig._first_task(response_data)
        if task.get("status") == "failed":
            raise ValueError(f"Kling video generation failed: {task.get('message') or response_data.get('message')}")
        outputs: Final = task.get("outputs")
        for output in outputs if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)) else ():
            if isinstance(output, Mapping) and output.get("type") == "video":
                url = output.get("url")
                if isinstance(url, str) and url:
                    return url
        raise ValueError("Video URL not found in Kling response. The job may still be processing.")

    @staticmethod
    def _coerce_start_image(input_reference: str | FileTypes | None) -> str | None:
        """
        A URL, unchanged.

        The path surface's `contents[].url` takes a URL and NOT base64 - unlike
        the classic `image` field - so a raw file is refused here instead of
        being encoded into a field the vendor would reject.
        """
        if input_reference is None:
            return None
        if isinstance(input_reference, str):
            stripped: Final = input_reference.strip()
            return stripped or None
        extract_file_data(input_reference)  # validates the payload before rejecting it by shape
        raise litellm.BadRequestError(
            message=(
                "The path-based Kling models take a start frame as a URL (contents[].url); an inline file or "
                "base64 payload has nowhere to go on this surface. Upload the image and pass its URL."
            ),
            model="kling",
            llm_provider=litellm.LlmProviders.KLING.value,
        )


__all__ = [  # mutable-ok: module export list, the shape Python expects
    "PATH_MODELS",
    "PATH_TASK_KIND",
    "KlingPathVideoConfig",
    "is_kling_path_model",
]  # mutable-ok: module export list, the shape Python expects
