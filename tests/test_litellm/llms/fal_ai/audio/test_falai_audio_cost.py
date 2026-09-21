import types

import pytest

import litellm
from litellm.cost_calculator import cost_per_token

ELEVEN_V3 = "fal_ai/fal-ai/elevenlabs/tts/eleven-v3"
ELEVEN_MUSIC = "fal_ai/fal-ai/elevenlabs/music"
LYRIA3_PRO = "fal_ai/fal-ai/lyria3/pro"


@pytest.fixture(autouse=True)
def _local_cost_map(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    yield


def test_char_priced_tts_uses_prompt_characters():
    prompt_cost, completion_cost = cost_per_token(
        model="fal_ai/fal-ai/elevenlabs/tts/eleven-v3",
        custom_llm_provider="fal_ai",
        call_type="speech",
        prompt_characters=1000,
    )
    assert prompt_cost == pytest.approx(0.0001 * 1000)
    assert completion_cost == 0.0


def test_flat_priced_music_returns_fixed_audio_cost():
    prompt_cost, completion_cost = cost_per_token(
        model="fal_ai/fal-ai/lyria3/pro",
        custom_llm_provider="fal_ai",
        call_type="speech",
    )
    assert prompt_cost == 0.0
    assert completion_cost == pytest.approx(0.08)


def test_per_second_music_multiplies_decoded_duration():
    # fal bills this app by the MINUTE at $0.60, so the per-second rate is
    # 0.60 / 60 = 0.01 and a 30 s track costs $0.30. The map used to carry
    # 0.013333 ($0.80/min), overstating every music COGS row by 33% (NOL-1098,
    # fal's admin billing API cross-checked against the proxy ledger).
    response = types.SimpleNamespace(
        _hidden_params={"audio_output_duration": 30.0}
    )
    prompt_cost, completion_cost = cost_per_token(
        model="fal_ai/fal-ai/elevenlabs/music",
        custom_llm_provider="fal_ai",
        call_type="speech",
        response=response,
    )
    assert prompt_cost == 0.0
    assert completion_cost == pytest.approx(0.01 * 30.0)
    assert completion_cost == pytest.approx(0.30)


def test_per_second_music_falls_back_to_zero_without_duration():
    prompt_cost, completion_cost = cost_per_token(
        model="fal_ai/fal-ai/elevenlabs/music",
        custom_llm_provider="fal_ai",
        call_type="speech",
        response=None,
    )
    assert prompt_cost == 0.0
    assert completion_cost == 0.0
