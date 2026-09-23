from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS

if TYPE_CHECKING:
    from litellm import Logging as LiteLLMLoggingObj
    from litellm.exceptions import ContentPolicyViolationError
    from litellm.types.llms.openai import ChatCompletionUserMessage, HttpxBinaryResponseContent
    from litellm.types.utils import ModelResponse


def _completion_response_cost(model_response: "ModelResponse") -> float | None:
    hidden_params: Final = getattr(model_response, "_hidden_params", None)
    if not isinstance(hidden_params, dict):
        return None
    response_cost: Final = hidden_params.get("response_cost")
    return response_cost if isinstance(response_cost, float) else None


GEMINI_TTS_CHAT_AUDIO_FORMAT: Final = "pcm16"
GEMINI_TTS_RAW_RESPONSE_FORMAT: Final = "pcm"
GEMINI_TTS_SUPPORTED_RESPONSE_FORMATS: Final = frozenset({"wav", GEMINI_TTS_RAW_RESPONSE_FORMAT})

# The Content-Type used when the bytes match no container we recognise. It is
# what this bridge returned unconditionally before NOL-1099, so an unknown
# container degrades to the old behaviour rather than to an error.
DEFAULT_AUDIO_CONTENT_TYPE: Final = "audio/mpeg"


def _sniff_audio_content_type(audio: bytes) -> str:
    """The container ``audio`` actually is, from its own header bytes.

    WHY SNIFF RATHER THAN TRUST THE PROVIDER'S DECLARED TYPE (NOL-1099). This
    bridge used to answer ``audio/mpeg`` for every non-Gemini-TTS model,
    because the only provider reaching it returned MP3. Gemini's declared
    ``inlineData.mimeType`` never survives the trip: the chat transform
    (``VertexGeminiConfig._extract_audio_response_from_parts``) drops it when
    it builds ``ChatCompletionAudioResponse``, which has no field to carry it,
    and that object is the OpenAI-shaped type the chat completions API already
    returns to callers — widening it to carry a mime type would change a public
    response shape to fix a private one.

    The bytes, however, are right here and cannot disagree with themselves. So
    the container is read off the header, which is both simpler and strictly
    more trustworthy than a declared type: it stays correct if a provider
    mislabels its own output, and it needs no per-provider plumbing.

    THIS MATTERS BECAUSE THE CALLER BELIEVES US. nolgia-api takes the
    Content-Type from this response and uses it for the stored object's type
    AND for the file extension. A wrong header there does not fail — it ships
    a playable-looking file that no player can open, after the customer has
    been billed. Lyria happens to return MP3, so the old hardcode was correct
    by luck; this removes the luck.
    """
    if audio[:3] == b"ID3":
        return "audio/mpeg"
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return "audio/wav"
    if audio[:4] == b"OggS":
        return "audio/ogg"
    if audio[:4] == b"fLaC":
        return "audio/flac"
    # ISO-BMFF (m4a/mp4 audio): the brand box starts at offset 4.
    if audio[4:8] == b"ftyp":
        return "audio/mp4"
    if len(audio) >= 2 and audio[0] == 0xFF:
        # AAC ADTS is 0xFFF1/0xFFF9 and must be tested BEFORE the generic MPEG
        # frame sync below, whose 0xFFE0 mask also matches it.
        if audio[1] in (0xF1, 0xF9):
            return "audio/aac"
        if audio[1] & 0xE0 == 0xE0:
            return "audio/mpeg"
    return DEFAULT_AUDIO_CONTENT_TYPE


class _ContentFilterRefusal(NamedTuple):
    detail: str
    evidence_key: str
    evidence: object


def _content_filter_refusal(prompt_feedback: object, blocked_candidate: object) -> _ContentFilterRefusal:
    if isinstance(prompt_feedback, Mapping) and prompt_feedback.get("blockReason"):
        block_message: Final = prompt_feedback.get("blockReasonMessage")
        return _ContentFilterRefusal(
            "the provider refused this prompt under its content policy and generated nothing: "
            f"promptFeedback.blockReason={prompt_feedback.get('blockReason')}"
            + (f" ({block_message})" if block_message else ""),
            "promptFeedback",
            prompt_feedback,
        )
    if isinstance(blocked_candidate, Mapping) and blocked_candidate.get("finishReason"):
        finish_message: Final = blocked_candidate.get("finishMessage")
        return _ContentFilterRefusal(
            "the provider's content policy blocked the generated audio: "
            f"finishReason={blocked_candidate.get('finishReason')}"
            + (f" ({finish_message})" if finish_message else ""),
            "candidate",
            blocked_candidate,
        )
    return _ContentFilterRefusal("the provider's content policy filtered the response and returned no audio", "", None)


def _content_filter_error(
    model_response: "ModelResponse", model: str, custom_llm_provider: str
) -> "ContentPolicyViolationError":
    """A filtered completion has no audio part. Reading it anyway raised an AttributeError that surfaced as a
    retried 500 (NOL-1143), so a refusal is a non-retried 400 that carries the provider's own reason."""
    from litellm.exceptions import ContentPolicyViolationError

    hidden_params: Final[Mapping[str, object]] = getattr(model_response, "_hidden_params", None) or MappingProxyType({})
    refusal: Final = _content_filter_refusal(
        hidden_params.get("vertex_ai_prompt_feedback"), hidden_params.get("vertex_ai_blocked_candidate")
    )
    provider_fields: Final = {refusal.evidence_key: refusal.evidence}  # mutable-ok: the exception field is a dict
    return ContentPolicyViolationError(
        message=refusal.detail,
        model=model,
        llm_provider=custom_llm_provider,
        provider_specific_fields=provider_fields if refusal.evidence_key else None,
    )


class ChatAudioParam(TypedDict):
    voice: ReadOnly[str]
    format: ReadOnly[NotRequired[str]]


class SpeechToCompletionBridgeTransformationHandler:
    def _validate_response_format(
        self, model: str, custom_llm_provider: str, optional_params: Mapping[str, object]
    ) -> None:
        if not self._is_gemini_tts_model(model):
            return
        response_format: Final = optional_params.get("response_format")
        if not isinstance(response_format, str) or response_format in GEMINI_TTS_SUPPORTED_RESPONSE_FORMATS:
            return
        from litellm.exceptions import BadRequestError

        supported: Final = ", ".join(sorted(GEMINI_TTS_SUPPORTED_RESPONSE_FORMATS))
        raise BadRequestError(
            message=(
                f"Gemini TTS only produces raw PCM16 audio, so response_format='{response_format}'"
                f" is not supported. Supported response formats: {supported}."
            ),
            model=model,
            llm_provider=custom_llm_provider,
        )

    def _chat_completion_params(self, optional_params: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(
            {
                param: value
                for param, value in optional_params.items()
                if param in OPENAI_CHAT_COMPLETION_PARAMS and param != "response_format"
            }
        )

    def _chat_audio_format(self, model: str, optional_params: Mapping[str, object]) -> str | None:
        if self._is_gemini_tts_model(model):
            return GEMINI_TTS_CHAT_AUDIO_FORMAT
        response_format: Final = optional_params.get("response_format")
        return response_format if isinstance(response_format, str) else None

    def _chat_audio_param(
        self, model: str, voice: str | Mapping[str, object] | None, optional_params: Mapping[str, object]
    ) -> ChatAudioParam | None:
        if not isinstance(voice, str):
            return None
        audio_format: Final = self._chat_audio_format(model, optional_params)
        if audio_format is None:
            voice_only: Final[ChatAudioParam] = {"voice": voice}
            return voice_only
        audio: Final[ChatAudioParam] = {"voice": voice, "format": audio_format}
        return audio

    def transform_request(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        litellm_logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str,
    ) -> dict:
        self._validate_response_format(model, custom_llm_provider, optional_params)
        user_message: Final[ChatCompletionUserMessage] = {"role": "user", "content": input}
        return_kwargs: Final = {
            "model": model,
            "messages": [user_message],
            "modalities": ["audio"],
            **self._chat_completion_params(optional_params),
            "audio": self._chat_audio_param(model, voice, optional_params),
            **litellm_params,
            "headers": headers,
            "litellm_logging_obj": litellm_logging_obj,
            "custom_llm_provider": custom_llm_provider,
        }
        return {k: v for k, v in return_kwargs.items() if v is not None}

    def _convert_pcm16_to_wav(self, pcm_data: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
        """
        Convert raw PCM16 data to WAV format.

        Args:
            pcm_data: Raw PCM16 audio data
            sample_rate: Sample rate in Hz (Gemini TTS typically uses 24000)
            channels: Number of audio channels (1 for mono)

        Returns:
            bytes: WAV formatted audio data
        """
        import struct

        # WAV header parameters
        byte_rate: Final = sample_rate * channels * 2  # 2 bytes per sample (16-bit)
        block_align: Final = channels * 2
        data_size: Final = len(pcm_data)
        file_size: Final = 36 + data_size

        # Create WAV header
        wav_header: Final = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",  # Chunk ID
            file_size,  # Chunk Size
            b"WAVE",  # Format
            b"fmt ",  # Subchunk1 ID
            16,  # Subchunk1 Size (PCM)
            1,  # Audio Format (PCM)
            channels,  # Number of Channels
            sample_rate,  # Sample Rate
            byte_rate,  # Byte Rate
            block_align,  # Block Align
            16,  # Bits per Sample
            b"data",  # Subchunk2 ID
            data_size,  # Subchunk2 Size
        )

        return wav_header + pcm_data

    def _is_gemini_tts_model(self, model: str) -> bool:
        """Check if the model is a Gemini TTS model that returns PCM16 data."""
        return "gemini" in model.lower() and ("tts" in model.lower() or "preview-tts" in model.lower())

    def _gemini_tts_response_body(self, decoded_audio: bytes, response_format: str | None) -> tuple[bytes, str]:
        if response_format == GEMINI_TTS_RAW_RESPONSE_FORMAT:
            return decoded_audio, "audio/pcm"
        return self._convert_pcm16_to_wav(decoded_audio), "audio/wav"

    def transform_response(
        self, model_response: "ModelResponse", response_format: str | None, custom_llm_provider: str = "gemini"
    ) -> "HttpxBinaryResponseContent":
        import base64

        import httpx

        from litellm.types.llms.openai import HttpxBinaryResponseContent
        from litellm.types.utils import ChatCompletionAudioResponse, Choices

        model: Final = getattr(model_response, "model", "")
        choice: Final = cast(Choices, model_response.choices[0])
        audio_part: Final[ChatCompletionAudioResponse | None] = getattr(choice.message, "audio", None)
        if audio_part is None:
            if choice.finish_reason == "content_filter":
                raise _content_filter_error(model_response, model, custom_llm_provider)
            raise ValueError(f"No audio part found in the response (finish_reason={choice.finish_reason!r})")
        decoded_audio: Final = base64.b64decode(audio_part.data)

        content, content_type = (
            self._gemini_tts_response_body(decoded_audio, response_format)
            if self._is_gemini_tts_model(model)
            else (decoded_audio, _sniff_audio_content_type(decoded_audio))
        )
        response: Final = httpx.Response(
            status_code=200, content=content, headers=MappingProxyType({"Content-Type": content_type})
        )
        binary_response: Final = HttpxBinaryResponseContent(response)
        binary_response.set_response_cost(_completion_response_cost(model_response))
        return binary_response
