"""
Kling's path-based surface: 3.0 Turbo, 3.0 Omni, O1 (NOL-1041).

Every expectation here was measured against the live API on 2026-09-19, either
by the zero-cost schema walk (one field deliberately invalid, so nothing can
submit) or by one paid render per model; the per-second rates in the comments
came back from Kling's own meter, not from its published table.
"""

from typing import Final
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.kling.auth import kling_console_auth_headers, resolve_kling_console_api_key
from litellm.llms.kling.videos import (
    KlingPathVideoConfig,
    KlingVideoConfig,
    get_kling_video_config,
    is_kling_path_model,
)
from litellm.llms.kling.videos.path_transformation import PATH_TASK_KIND
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.videos.capabilities import DeclaredCapabilityParams

API_BASE: Final = "https://api-singapore.klingai.com"
CONSOLE_KEY: Final = "api-" + "z" * 53  # synthetic, not a real credential


def _create_response(payload, status_code=200):
    request = httpx.Request("POST", f"{API_BASE}/text-to-video/kling-3.0-turbo")
    return httpx.Response(status_code, json=payload, request=request)


def _tasks_response(payload, status_code=200):
    request = httpx.Request("GET", f"{API_BASE}/tasks?task_ids=930490170191773698")
    return httpx.Response(status_code, json=payload, request=request)


class TestPathModelSelection:
    @pytest.mark.parametrize(
        "model",
        ["kling/kling-3.0-turbo", "kling/kling-3.0-omni", "kling/kling-3.0-omni-audio", "kling/kling-o1"],
    )
    def test_path_models_select_the_path_config(self, model):
        assert isinstance(get_kling_video_config(model), KlingPathVideoConfig)

    @pytest.mark.parametrize("model", ["kling/kling-v3", "kling/kling-v2-6", "kling/kling-v3-motion-control", None])
    def test_classic_models_keep_the_classic_config(self, model):
        config = get_kling_video_config(model)
        assert isinstance(config, KlingVideoConfig)
        assert not isinstance(config, KlingPathVideoConfig)

    def test_encoded_task_kind_routes_back_to_the_path_config(self):
        """A status lookup resolves its config from the id's model_id, not the model name."""
        assert is_kling_path_model(PATH_TASK_KIND)
        assert isinstance(get_kling_video_config(PATH_TASK_KIND), KlingPathVideoConfig)

    @pytest.mark.parametrize("kind", ["text2video", "image2video", "motion-control", "avatar/image2video"])
    def test_classic_task_kinds_do_not_route_to_the_path_config(self, kind):
        assert not is_kling_path_model(kind)


class TestConsoleAuth:
    def test_bearer_is_the_console_key_verbatim(self):
        headers = kling_console_auth_headers(CONSOLE_KEY)
        assert headers["Authorization"] == f"Bearer {CONSOLE_KEY}"
        assert headers["Content-Type"] == "application/json"

    def test_an_ak_sk_pair_is_refused_rather_than_sent(self):
        """The path surface answers AK/SK with 401 code 1002; fail before the round trip."""
        with pytest.raises(ValueError, match="AccessKey:SecretKey"):
            resolve_kling_console_api_key("AK123:SK456")

    def test_missing_key_names_the_env_var(self, monkeypatch):
        monkeypatch.delenv("KLING_CONSOLE_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None)
        with pytest.raises(ValueError, match="KLING_CONSOLE_API_KEY"):
            resolve_kling_console_api_key(None)

    def test_does_not_fall_back_to_the_classic_credential(self, monkeypatch):
        """litellm.api_key may hold the classic AK:SK pair; it must not leak here."""
        monkeypatch.delenv("KLING_CONSOLE_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", "AK:SK")
        with pytest.raises(ValueError, match="KLING_CONSOLE_API_KEY"):
            resolve_kling_console_api_key(None)

    def test_env_var_is_read(self, monkeypatch):
        monkeypatch.setenv("KLING_CONSOLE_API_KEY", CONSOLE_KEY)
        assert resolve_kling_console_api_key(None) == CONSOLE_KEY


class TestPathRequests:
    def setup_method(self):
        self.config = KlingPathVideoConfig()
        self.logging_obj = Mock()
        self.headers = {"Authorization": f"Bearer {CONSOLE_KEY}"}

    def _build(self, model, prompt="a red ceramic cube on a turntable", **params):
        mapped = self.config.map_openai_params(params, model, False)
        return self.config.transform_video_create_request(
            model, prompt, API_BASE, mapped, GenericLiteLLMParams(), self.headers
        )

    def test_base_url_drops_the_v1_prefix(self):
        assert self.config.get_complete_url("kling/kling-o1", None, {}) == API_BASE

    def test_turbo_text_to_video_is_prompt_shaped(self):
        body, _, url = self._build("kling/kling-3.0-turbo", seconds=5, resolution="720p")
        assert url == f"{API_BASE}/text-to-video/kling-3.0-turbo"
        assert body["prompt"] == "a red ceramic cube on a turntable"
        assert "contents" not in body
        assert body["settings"] == {"resolution": "720p", "duration": 5}

    def test_turbo_image_to_video_switches_path_and_shape(self):
        body, _, url = self._build(
            "kling/kling-3.0-turbo", seconds=5, resolution="1080p", input_reference="https://x.test/a.jpg"
        )
        assert url == f"{API_BASE}/image-to-video/kling-3.0-turbo"
        assert body["contents"] == [
            {"type": "prompt", "text": "a red ceramic cube on a turntable"},
            {"type": "first_frame", "url": "https://x.test/a.jpg"},
        ]
        assert "prompt" not in body

    def test_turbo_never_sends_an_audio_setting(self):
        """Kling publishes only a native-audio rate for Turbo, so there is nothing to toggle."""
        body, _, _ = self._build("kling/kling-3.0-turbo", seconds=5)
        assert "audio" not in body["settings"]

    @pytest.mark.parametrize(
        ("model", "expected_audio"),
        [("kling/kling-3.0-omni", "off"), ("kling/kling-3.0-omni-audio", "native"), ("kling/kling-o1", "off")],
    )
    def test_audio_is_pinned_by_model_id(self, model, expected_audio):
        """One id per published rate row: the transform always sends the value that id is priced for."""
        body, _, _ = self._build(model, seconds=5)
        assert body["settings"]["audio"] == expected_audio

    def test_the_two_omni_ids_share_one_endpoint(self):
        _, _, silent = self._build("kling/kling-3.0-omni", seconds=5)
        _, _, audible = self._build("kling/kling-3.0-omni-audio", seconds=5)
        assert silent == audible == f"{API_BASE}/omni-video/kling-3.0-omni"

    def test_omni_is_contents_shaped_without_an_image(self):
        body, _, url = self._build("kling/kling-o1", seconds=5)
        assert url == f"{API_BASE}/omni-video/kling-o1"
        assert body["contents"] == [{"type": "prompt", "text": "a red ceramic cube on a turntable"}]

    def test_size_becomes_an_aspect_ratio_setting(self):
        body, _, _ = self._build("kling/kling-3.0-turbo", seconds=5, size="720x1280")
        assert body["settings"]["aspect_ratio"] == "9:16"

    def test_options_are_nested_and_omitted_when_empty(self):
        plain, _, _ = self._build("kling/kling-o1", seconds=5)
        assert "options" not in plain
        with_options, _, _ = self._build("kling/kling-o1", seconds=5, external_task_id="abc")
        assert with_options["options"] == {"external_task_id": "abc"}

    @pytest.mark.parametrize(
        ("model", "image"),
        [
            ("kling/kling-3.0-turbo", None),
            ("kling/kling-3.0-turbo", "https://x.test/a.jpg"),
            ("kling/kling-3.0-omni", None),
            ("kling/kling-3.0-omni-audio", None),
            ("kling/kling-o1", None),
        ],
    )
    def test_callback_url_rides_options_on_every_path_endpoint(self, model, image):
        """A caller's callback_url reaches the vendor under options, never settings (NOL-1042).

        nolgia-api sends it as a top-level request param, exactly as external_task_id
        travels; the path surface accepts it on every endpoint (measured NOL-1041).
        """
        callback: Final = "https://api.nolgia.ai/v1/callbacks/kling?token=v1.abc.def"
        body, _, _ = self._build(
            model, seconds=5, callback_url=callback, **({"input_reference": image} if image else {})
        )
        assert body["options"] == {"callback_url": callback}
        assert "callback_url" not in body["settings"]
        assert "callback_url" not in body

    def test_default_duration_and_resolution_are_the_cheapest_priced_rungs(self):
        body, _, _ = self._build("kling/kling-o1")
        assert body["settings"]["duration"] == 5
        assert body["settings"]["resolution"] == "720p"


class TestPathRefusals:
    def setup_method(self):
        self.config = KlingPathVideoConfig()

    @pytest.mark.parametrize("model", ["kling/kling-3.0-turbo", "kling/kling-o1"])
    def test_4k_is_refused_where_the_vendor_publishes_no_rate(self, model):
        """Measured: both endpoints answer 'video resolution value 4k is invalid'."""
        with pytest.raises(litellm.BadRequestError, match="does not publish a '4k' rate"):
            self.config.map_openai_params({"seconds": 5, "resolution": "4k"}, model, False)

    def test_o1_duration_stops_at_ten(self):
        """Measured: duration 11 is rejected by the vendor, 10 is accepted."""
        self.config.map_openai_params({"seconds": 10}, "kling/kling-o1", False)
        with pytest.raises(litellm.BadRequestError, match="3 through 10"):
            self.config.map_openai_params({"seconds": 11}, "kling/kling-o1", False)

    def test_omni_duration_runs_to_fifteen(self):
        self.config.map_openai_params({"seconds": 15}, "kling/kling-3.0-omni", False)
        with pytest.raises(litellm.BadRequestError, match="3 through 15"):
            self.config.map_openai_params({"seconds": 16}, "kling/kling-3.0-omni", False)

    @pytest.mark.parametrize("seconds", [5.5, "abc", True, None.__class__])
    def test_non_integer_durations_are_refused(self, seconds):
        with pytest.raises(litellm.BadRequestError):
            self.config.map_openai_params({"seconds": seconds}, "kling/kling-o1", False)

    def test_per_request_audio_is_refused_and_names_the_priced_id(self):
        with pytest.raises(litellm.BadRequestError, match=r"kling-3\.0-omni-audio"):
            self.config.map_openai_params({"seconds": 5, "generate_audio": True}, "kling/kling-3.0-omni", False)

    def test_generate_audio_is_not_advertised(self):
        support = self.config.get_capability_param_support("kling/kling-3.0-omni-audio")
        assert support == DeclaredCapabilityParams(frozenset(("input_reference", "image_url")))
        assert "generate_audio" not in self.config.get_supported_openai_params("kling/kling-3.0-omni-audio")

    def test_a_classic_model_id_is_refused_by_the_path_config(self):
        with pytest.raises(litellm.BadRequestError, match="not a path-based Kling model"):
            self.config.map_openai_params({"seconds": 5}, "kling/kling-v3", False)

    def test_an_inline_image_is_refused_because_the_field_takes_a_url(self):
        """contents[].url takes a URL; the classic surface's base64 `image` field has no analogue here."""
        with pytest.raises(litellm.BadRequestError, match="pass its URL"):
            self.config.map_openai_params(
                {"seconds": 5, "input_reference": ("frame.png", b"\x89PNG\r\n\x1a\n", "image/png")},
                "kling/kling-3.0-turbo",
                False,
            )

    def test_an_empty_request_is_refused_before_the_round_trip(self):
        mapped = self.config.map_openai_params({"seconds": 5}, "kling/kling-o1", False)
        with pytest.raises(litellm.BadRequestError, match="requires a prompt"):
            self.config.transform_video_create_request(
                "kling/kling-o1", "   ", API_BASE, mapped, GenericLiteLLMParams(), {}
            )


class TestPathResponses:
    def setup_method(self):
        self.config = KlingPathVideoConfig()
        self.logging_obj = Mock()

    def _created(self, **settings):
        request_data = {"settings": {"resolution": "720p", "duration": 5, **settings}}
        return self.config.transform_video_create_response(
            model="kling/kling-3.0-turbo",
            raw_response=_create_response(
                {"code": 0, "message": "SUCCEED", "data": {"id": "930490170191773698", "status": "submitted"}}
            ),
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
            request_data=request_data,
        )

    def test_create_reads_data_id_not_data_task_id(self):
        """The path surface renamed the field; reading task_id would strand every job."""
        video = self._created()
        assert decode_video_id_with_provider(video.id)["video_id"] == "930490170191773698"
        assert video.status == "queued"

    def test_cogs_basis_is_the_requested_duration_and_the_resolution_tier(self):
        """Measured: a 5 s request returned a 5.041 s file and billed exactly 5x the rate."""
        video = self._created()
        assert video.usage["duration_seconds"] == 5.0
        assert video.usage["video_resolution"] == "720p"

    def test_the_encoded_kind_routes_a_later_lookup_back_here(self):
        video = self._created()
        assert decode_video_id_with_provider(video.id)["model_id"] == PATH_TASK_KIND

    def test_a_vendor_body_code_becomes_an_error_under_http_200(self):
        """Kling reports field errors in the BODY under HTTP 200, so the code is the only signal."""
        with pytest.raises(BaseLLMException, match="duration value"):
            self.config.transform_video_create_response(
                model="kling/kling-o1",
                raw_response=_create_response({"code": 1201, "message": "duration value '99' is invalid"}),
                logging_obj=self.logging_obj,
                custom_llm_provider="kling",
                request_data={"settings": {}},
            )

    def test_status_reads_the_first_element_of_a_list(self):
        """GET /tasks answers with a LIST under data, even for a single id."""
        response = self.config.transform_video_status_retrieve_response(
            raw_response=_tasks_response(
                {"code": 0, "data": [{"id": "930490170191773698", "status": "succeeded", "outputs": []}]}
            ),
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
        )
        assert response.status == "completed"

    def test_a_failed_task_carries_the_vendor_message(self):
        response = self.config.transform_video_status_retrieve_response(
            raw_response=_tasks_response({"code": 0, "data": [{"id": "1", "status": "failed", "message": "risk"}]}),
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
        )
        assert response.status == "failed"
        assert response.error["message"] == "risk"

    def test_an_empty_task_list_degrades_to_queued(self):
        response = self.config.transform_video_status_retrieve_response(
            raw_response=_tasks_response({"code": 0, "data": []}),
            logging_obj=self.logging_obj,
            custom_llm_provider="kling",
        )
        assert response.status == "queued"

    def test_content_url_comes_from_the_video_output(self):
        payload = {
            "code": 0,
            "data": [
                {
                    "id": "1",
                    "status": "succeeded",
                    "outputs": [
                        {"type": "image", "url": "https://x.test/poster.jpg"},
                        {"type": "video", "url": "https://x.test/clip.mp4", "duration": "5.041"},
                    ],
                }
            ],
        }
        assert self.config._extract_video_url(payload) == "https://x.test/clip.mp4"

    def test_a_still_processing_task_raises_rather_than_returning_a_poster(self):
        payload = {"code": 0, "data": [{"id": "1", "status": "processing", "outputs": []}]}
        with pytest.raises(ValueError, match="may still be processing"):
            self.config._extract_video_url(payload)


class TestPathTaskUrl:
    def setup_method(self):
        self.config = KlingPathVideoConfig()

    def test_poll_url_is_the_shared_tasks_query(self):
        encoded = encode_video_id_with_provider("930490170191773698", "kling", PATH_TASK_KIND)
        url, _ = self.config.transform_video_status_retrieve_request(encoded, API_BASE, GenericLiteLLMParams(), {})
        assert url == f"{API_BASE}/tasks?task_ids=930490170191773698"

    def test_a_non_numeric_task_id_is_refused(self):
        """Kling task ids are decimal snowflakes; anything else did not come from a create."""
        encoded = encode_video_id_with_provider("../../admin", "kling", PATH_TASK_KIND)
        with pytest.raises(ValueError, match="must be numeric"):
            self.config.transform_video_status_retrieve_request(encoded, API_BASE, GenericLiteLLMParams(), {})
