import base64
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.constants import OPENAI_CHAT_COMPLETION_PARAMS
from litellm.endpoints.speech.speech_to_completion_bridge.transformation import (
    SpeechToCompletionBridgeTransformationHandler,
    _sniff_audio_content_type,
)
from litellm.types.utils import ChatCompletionAudioResponse, Choices, Message, ModelResponse

GEMINI_TTS_MODEL: Final = "gemini-3.1-flash-tts-preview"
PCM_BYTES: Final = b"\x01\x02\x03\x04" * 6


def _model_response(model: str, pcm: bytes) -> ModelResponse:
    audio: Final = ChatCompletionAudioResponse(data=base64.b64encode(pcm).decode(), expires_at=0, transcript="hello")
    return ModelResponse(model=model, choices=[Choices(message=Message(content=None, audio=audio))])


def _bridge_request(response_format: str | None) -> dict:
    optional_params: Final = (
        {"temperature": 0.4} if response_format is None else {"temperature": 0.4, "response_format": response_format}
    )
    return SpeechToCompletionBridgeTransformationHandler().transform_request(
        model=GEMINI_TTS_MODEL,
        input="Hello from LiteLLM",
        voice="Kore",
        optional_params=optional_params,
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="gemini",
    )


@pytest.mark.parametrize("response_format", ["wav", "pcm", None])
def test_gemini_tts_request_keeps_speech_response_format_out_of_chat_params(response_format: str | None) -> None:
    request: Final = _bridge_request(response_format)

    assert "response_format" not in request
    assert request["audio"] == {"voice": "Kore", "format": "pcm16"}
    assert request["temperature"] == 0.4
    assert request["modalities"] == ["audio"]

    gemini_params: Final = litellm.get_optional_params(
        model=GEMINI_TTS_MODEL,
        custom_llm_provider="gemini",
        **{param: value for param, value in request.items() if param in OPENAI_CHAT_COMPLETION_PARAMS},
    )
    assert gemini_params["speechConfig"] == {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
    assert "responseMimeType" not in gemini_params


def test_non_gemini_request_forwards_speech_response_format_as_audio_format() -> None:
    request: Final = SpeechToCompletionBridgeTransformationHandler().transform_request(
        model="gpt-4o-audio-preview",
        input="Hello from LiteLLM",
        voice="alloy",
        optional_params={"response_format": "wav"},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )

    assert "response_format" not in request
    assert request["audio"] == {"voice": "alloy", "format": "wav"}


@pytest.mark.parametrize("response_format", ["mp3", "flac", "opus", "aac"])
def test_gemini_tts_request_rejects_formats_gemini_cannot_produce(response_format: str) -> None:
    with pytest.raises(litellm.BadRequestError) as excinfo:
        _bridge_request(response_format)

    assert excinfo.value.status_code == 400
    assert response_format in str(excinfo.value)
    assert "pcm" in str(excinfo.value)
    assert "wav" in str(excinfo.value)


def test_gemini_tts_pcm_response_returns_raw_pcm_bytes() -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(GEMINI_TTS_MODEL, PCM_BYTES),
        response_format="pcm",
    )

    assert response.response.content == PCM_BYTES
    assert response.response.headers["content-type"] == "audio/pcm"


@pytest.mark.parametrize("response_format", ["wav", None])
def test_gemini_tts_wav_and_default_responses_wrap_pcm_in_wav(response_format: str | None) -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(GEMINI_TTS_MODEL, PCM_BYTES),
        response_format=response_format,
    )

    body: Final = response.response.content
    assert body[:4] == b"RIFF"
    assert body[8:12] == b"WAVE"
    assert body[44:] == PCM_BYTES
    assert response.response.headers["content-type"] == "audio/wav"


def test_non_gemini_response_keeps_original_bytes_and_mpeg_content_type() -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response("gpt-4o-audio-preview", PCM_BYTES),
        response_format="mp3",
    )

    assert response.response.content == PCM_BYTES
    assert response.response.headers["content-type"] == "audio/mpeg"


# --- NOL-1099: the container a non-TTS Gemini model actually returned --------
#
# The bridge answered "audio/mpeg" for every non-Gemini-TTS model because the
# only provider reaching it returned MP3. Lyria (Google's music model, moved
# off the fal reseller onto our own Gemini key) is the first that does not
# have "tts" in its name, and nolgia-api derives both the stored content type
# and the FILE EXTENSION from this header — so a wrong answer here ships a
# file no player can open, after the customer has been billed.

LYRIA_MODEL: Final = "gemini/lyria-3.5"

# Real header bytes. The MP3 case is what production Lyria actually returns:
# a `generateContent` call from our egress on 2026-09-21 came back with
# inlineData.mimeType "audio/mpeg" and bytes beginning "ID3" (ffprobe: mp3,
# 44100 Hz, stereo, 192 kbps, 175.4 s).
_ID3_MP3: Final = b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 16
_SYNC_MP3: Final = b"\xff\xfb\x90\x00" + b"\x00" * 16
_WAV: Final = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 16
_OGG: Final = b"OggS\x00\x02" + b"\x00" * 20
_FLAC: Final = b"fLaC\x00\x00\x00\x22" + b"\x00" * 16
_M4A: Final = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 16
_AAC: Final = b"\xff\xf1\x50\x80" + b"\x00" * 16
_UNKNOWN: Final = b"\x01\x02\x03\x04" * 6


@pytest.mark.parametrize(
    ("audio_bytes", "expected_content_type"),
    [
        (_ID3_MP3, "audio/mpeg"),
        (_SYNC_MP3, "audio/mpeg"),
        (_WAV, "audio/wav"),
        (_OGG, "audio/ogg"),
        (_FLAC, "audio/flac"),
        (_M4A, "audio/mp4"),
        (_AAC, "audio/aac"),
    ],
)
def test_non_tts_gemini_response_reports_the_container_it_actually_is(
    audio_bytes: bytes, expected_content_type: str
) -> None:
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(LYRIA_MODEL, audio_bytes), response_format=None
    )

    assert response.response.headers["Content-Type"] == expected_content_type
    # The bytes are passed through untouched — only the label is corrected.
    assert response.response.content == audio_bytes


def test_unrecognised_container_keeps_the_previous_default() -> None:
    """An unknown header degrades to the old hardcode, never to an error.

    A model whose container we cannot name is still a generation the customer
    has paid for; refusing it would turn a labelling gap into a failed request.
    """
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(LYRIA_MODEL, _UNKNOWN), response_format=None
    )

    assert response.response.headers["Content-Type"] == "audio/mpeg"
    assert response.response.content == _UNKNOWN


def test_aac_is_not_mistaken_for_mpeg() -> None:
    """ADTS AAC starts 0xFFF1/0xFFF9, which the 0xFFE0 MPEG frame-sync mask
    also matches. Order of the checks is the whole test."""
    assert _sniff_audio_content_type(_AAC) == "audio/aac"
    assert _sniff_audio_content_type(_SYNC_MP3) == "audio/mpeg"


def test_gemini_tts_path_is_unchanged_by_the_sniff() -> None:
    """Gemini TTS returns raw PCM with no header to sniff, and keeps its own
    PCM->WAV wrapping. This pins that NOL-1099 did not touch it."""
    response: Final = SpeechToCompletionBridgeTransformationHandler().transform_response(
        model_response=_model_response(GEMINI_TTS_MODEL, PCM_BYTES), response_format=None
    )

    assert response.response.headers["Content-Type"] == "audio/wav"
    assert response.response.content.startswith(b"RIFF")


def test_lyria_prices_as_a_flat_per_song_audio_generation() -> None:
    """NOL-1099: Lyria bills $0.08 per song, not per token.

    The route cannot be priced from litellm-config: `output_cost_per_audio` is
    in CustomPricingLiteLLMParams, so `shared_backend_model_info` strips it
    from the shared `{provider}/{model}` key that the speech cost path reads,
    and a deployment-level pin is inert. The BUNDLED PRICE MAP is the fix, and
    is what this pins — deliberately reading the vendored file rather than
    whatever `litellm.model_cost` happens to hold, because the proxy runs with
    `LITELLM_LOCAL_MODEL_COST_MAP=True` and that file is its only price source.
    Without these rows the route logs $0.00 — the NOL-535/559 class.
    """
    import json
    from pathlib import Path as _Path

    from litellm.cost_calculator import cost_per_token

    bundled: Final = json.loads(
        (_Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json").read_text()
    )

    expected: Final = {
        "gemini/lyria-3.5": 0.08,
        "gemini/lyria-3-pro-preview": 0.08,
        "gemini/lyria-3-clip-preview": 0.04,
    }

    for model, price in expected.items():
        row = bundled.get(model)
        assert row is not None, f"{model} is missing from the bundled price map"
        assert row["mode"] == "audio_speech", f"{model} must price on the /v1/audio/speech path"
        assert row["output_cost_per_audio"] == price, model
        # Stale per-token keys are what made this meter $0: select_cost_metric_for_model
        # reads them, and a zero there wins over having no per-song price at all.
        assert "input_cost_per_token" not in row, f"{model} still carries a per-token price"
        assert "output_cost_per_token" not in row, f"{model} still carries a per-token price"

    litellm.register_model(model_cost={model: bundled[model] for model in expected})

    for model, price in expected.items():
        prompt_cost, completion_cost = cost_per_token(
            model=model,
            custom_llm_provider="gemini",
            call_type="aspeech",
            prompt_tokens=21,
            completion_tokens=719,
            prompt_characters=83,
        )
        assert prompt_cost + completion_cost == pytest.approx(price), model


LYRIA_BLOCKED_PROMPT_RESPONSE: Final = {
    "promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},
    "usageMetadata": {
        "promptTokenCount": 20,
        "totalTokenCount": 20,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 20}],
        "serviceTier": "standard",
    },
    "modelVersion": "lyria-3.5",
    "responseId": "yg60arLwGPTUz7IP8o7yqAs",
}


@pytest.mark.asyncio
async def test_lyria_blocked_prompt_is_a_non_retried_content_policy_400(
    respx_mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    generate_content: Final = respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/lyria-3\.5:generateContent.*"
    ).mock(return_value=httpx.Response(200, json=LYRIA_BLOCKED_PROMPT_RESPONSE))
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "music-lyria3-pro",
                "litellm_params": {"model": "gemini/lyria-3.5", "api_key": "fake-gemini-key"},
                "model_info": {"mode": "audio_speech"},
            }
        ],
        num_retries=2,
    )

    with pytest.raises(litellm.ContentPolicyViolationError) as excinfo:
        await router.aspeech(model="music-lyria3-pro", input="a song", voice=None)

    assert excinfo.value.status_code == 400
    assert "promptFeedback.blockReason=PROHIBITED_CONTENT" in str(excinfo.value)
    assert "content policy" in str(excinfo.value)
    assert excinfo.value.provider_specific_fields == {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}
    assert generate_content.call_count == 1


def test_candidate_level_block_names_the_finish_reason() -> None:
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig

    blocked: Final = {
        "candidates": [{"finishReason": "PROHIBITED_CONTENT", "finishMessage": "Blocked.", "index": 0}],
        "usageMetadata": {"promptTokenCount": 6, "totalTokenCount": 6},
    }
    model_response: Final = VertexGeminiConfig()._transform_google_generate_content_to_openai_model_response(
        completion_response=blocked,
        model_response=ModelResponse(),
        model="lyria-3.5",
        logging_obj=MagicMock(optional_params={}),
        raw_response=MagicMock(),
    )

    with pytest.raises(litellm.ContentPolicyViolationError) as excinfo:
        SpeechToCompletionBridgeTransformationHandler().transform_response(model_response, None)

    assert "finishReason=PROHIBITED_CONTENT (Blocked.)" in str(excinfo.value)


def test_missing_audio_without_a_filter_is_not_reported_as_a_refusal() -> None:
    model_response: Final = ModelResponse(
        model="lyria-3.5", choices=[Choices(finish_reason="stop", message=Message(content="no song today"))]
    )

    with pytest.raises(ValueError, match="finish_reason='stop'"):
        SpeechToCompletionBridgeTransformationHandler().transform_response(model_response, None)
