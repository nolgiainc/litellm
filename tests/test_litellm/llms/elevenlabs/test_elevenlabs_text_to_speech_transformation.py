from typing import Final

import pytest

from litellm.llms.elevenlabs.text_to_speech.transformation import (
    ElevenLabsTextToSpeechConfig,
)


@pytest.mark.parametrize("existing_settings", [False, True])
def test_speed_is_forwarded_in_voice_settings(existing_settings: bool) -> None:
    config: Final = ElevenLabsTextToSpeechConfig()
    voice, params = config.map_openai_params(
        model="elevenlabs/tts",
        voice="alloy",
        optional_params={
            "speed": 1.1,
            **({"voice_settings": {"stability": 0.5, "speed": 1.0}} if existing_settings else {}),
        },
    )
    request: Final = config.transform_text_to_speech_request(
        model="elevenlabs/tts", input="hello", voice=voice, optional_params=params, litellm_params={}, headers={}
    )
    assert request["dict_body"]["voice_settings"] == {"speed": 1.1, **({"stability": 0.5} if existing_settings else {})}
    assert "speed" not in request["dict_body"]


def test_should_encode_elevenlabs_voice_id_path_segment():
    config = ElevenLabsTextToSpeechConfig()

    url = config.get_complete_url(
        model="elevenlabs/tts",
        api_base="https://api.elevenlabs.io",
        litellm_params={
            config.ELEVENLABS_VOICE_ID_KEY: "voice/../../models?x=1#frag",
        },
    )

    assert (
        url
        == "https://api.elevenlabs.io/v1/text-to-speech/voice%2F..%2F..%2Fmodels%3Fx%3D1%23frag"
    )


def test_should_reject_dot_segment_elevenlabs_voice_id():
    config = ElevenLabsTextToSpeechConfig()

    with pytest.raises(ValueError, match="voice_id cannot be a dot path segment"):
        config.get_complete_url(
            model="elevenlabs/tts",
            api_base="https://api.elevenlabs.io",
            litellm_params={config.ELEVENLABS_VOICE_ID_KEY: ".."},
        )
