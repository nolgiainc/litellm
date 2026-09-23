import asyncio
import time
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, Final, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import FAL_AI_DEFAULT_API_BASE, FAL_AI_POLLING_TIMEOUT
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client, get_async_httpx_client
from litellm.llms.fal_ai.utils import normalize_fal_model_id
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.llms.openai import (
        HttpxBinaryResponseContent as _HttpxBinaryResponseContent,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    HttpxBinaryResponseContent = _HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


_TERMINAL_OK = "COMPLETED"
_TERMINAL_FAIL = {"FAILED", "CANCELLED"}
_POLL_INTERVAL_SECS = 1.5


class FalAIAudioConfig(BaseTextToSpeechConfig):
    """
    fal.ai audio (TTS / music / SFX) via its queue API: submit goes through the
    shared BaseLLMHTTPHandler, then transform_text_to_speech_response polls the
    queue and downloads the rendered audio.
    """

    def __init__(self) -> None:
        super().__init__()
        self._polling_timeout_secs: float = float(FAL_AI_POLLING_TIMEOUT)

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "input",
            "voice",
            "response_format",
            "speed",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict = {},
    ) -> tuple[str | None, dict]:
        return (voice if isinstance(voice, str) else None), dict(optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_key = api_key or litellm.api_key or get_secret_str("FAL_AI_API_KEY") or get_secret_str("FAL_KEY")
        if not resolved_key:
            raise ValueError(
                "fal.ai API key is required. Set FAL_AI_API_KEY (or FAL_KEY) "
                "environment variable or pass api_key parameter."
            )
        headers.update(
            {
                "Authorization": f"Key {resolved_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base = api_base or get_secret_str("FAL_AI_API_BASE") or FAL_AI_DEFAULT_API_BASE
        model_id = normalize_fal_model_id(model)
        return f"{base.rstrip('/')}/{model_id}"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        body: dict[str, Any] = {"text": input, "prompt": input}
        if voice is not None:
            body["voice"] = voice
        for key, value in optional_params.items():
            if key in ("response_format", "speed", "extra_headers", "extra_body"):
                continue
            body[key] = value
        extra_body = optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            body.update(extra_body)
        request: Final[TextToSpeechRequestData] = {"dict_body": body, "headers": {}}
        if "speed" not in optional_params:
            return request
        model_id: Final = normalize_fal_model_id(model)
        if not model_id.startswith(("fal-ai/kokoro/", "fal-ai/minimax/speech-")):
            return request
        try:
            speed: Final = float(optional_params["speed"])
        except (TypeError, ValueError):
            return request
        if model_id.startswith("fal-ai/kokoro/"):
            kokoro_request: Final[TextToSpeechRequestData] = {"dict_body": {**body, "speed": speed}, "headers": {}}
            return kokoro_request
        voice_setting: Final = body.get("voice_setting")
        minimax_request: Final[TextToSpeechRequestData] = {
            "dict_body": {
                **body,
                "voice_setting": {
                    **(voice_setting if isinstance(voice_setting, dict) else {}),
                    "speed": max(0.5, min(2.0, speed)),
                },
            },
            "headers": {},
        }
        return minimax_request

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params_dict: dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: float | httpx.Timeout,
        extra_headers: dict[str, Any] | None,
        base_llm_http_handler: Any,
        aspeech: bool,
        api_base: str | None,
        api_key: str | None,
        **kwargs: Any,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[Any, Any, "HttpxBinaryResponseContent"],
    ]:
        api_base = (
            api_base
            or litellm_params_dict.get("api_base")
            or litellm.api_base
            or get_secret_str("FAL_AI_API_BASE")
            or FAL_AI_DEFAULT_API_BASE
        )
        api_key = (
            api_key
            or litellm_params_dict.get("api_key")
            or litellm.api_key
            or get_secret_str("FAL_AI_API_KEY")
            or get_secret_str("FAL_KEY")
        )
        litellm_params_dict.update({"api_key": api_key, "api_base": api_base})

        self._polling_timeout_secs = self._resolve_polling_timeout(timeout)

        merged_params = dict(optional_params)
        if "extra_body" not in merged_params and kwargs.get("extra_body") is not None:
            merged_params["extra_body"] = kwargs["extra_body"]

        voice_param = voice if isinstance(voice, str) else None

        return base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_param,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=merged_params,
            custom_llm_provider="fal_ai",
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
            client=None,
            _is_async=aspeech,
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        status_url, response_url = self._queue_urls(raw_response.json())
        headers: Final = self._poll_headers(raw_response)
        client: Final = _get_httpx_client()

        self._poll_until_complete_sync(
            status_url=status_url,
            headers=headers,
            client=client,
            timeout_secs=self._polling_timeout_secs,
        )

        result_resp: Final = client.get(url=response_url, headers=headers)
        self._raise_for_fal_status(result_resp)
        return self._binary_audio(client.get(url=self._extract_audio_url(result_resp.json())))

    async def async_transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        status_url, response_url = self._queue_urls(raw_response.json())
        headers: Final = self._poll_headers(raw_response)
        client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.FAL_AI)
        timeout_secs: Final = self._polling_timeout_secs
        deadline: Final = time.monotonic() + timeout_secs
        while not self._job_completed(await client.get(url=status_url, headers=headers), deadline, timeout_secs):
            await asyncio.sleep(_POLL_INTERVAL_SECS)

        result_resp: Final = await client.get(url=response_url, headers=headers)
        self._raise_for_fal_status(result_resp)
        return self._binary_audio(await client.get(url=self._extract_audio_url(result_resp.json())))

    @staticmethod
    def _binary_audio(binary_resp: httpx.Response) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        binary_resp.raise_for_status()
        result: Final = HttpxBinaryResponseContent(response=binary_resp)
        duration: Final = calculate_request_duration(binary_resp.content)
        if duration is not None:
            result._hidden_params = {"audio_output_duration": duration}
        return result

    def _raise_for_fal_status(self, resp: httpx.Response) -> None:
        if 400 <= resp.status_code < 500:
            raise self.get_error_class(
                error_message=resp.text,
                status_code=resp.status_code,
                headers=dict(resp.headers),  # mutable-ok: BaseTextToSpeechConfig.get_error_class requires a dict
            )
        resp.raise_for_status()

    @staticmethod
    def _resolve_polling_timeout(timeout: float | httpx.Timeout) -> float:
        candidate: Any = timeout
        if isinstance(timeout, httpx.Timeout):
            candidate = timeout.read or timeout.connect
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            return float(FAL_AI_POLLING_TIMEOUT)
        return value if value > 0 else float(FAL_AI_POLLING_TIMEOUT)

    @staticmethod
    def _queue_urls(submit_payload: dict[str, Any]) -> tuple[str, str]:
        status_url = submit_payload.get("status_url")
        response_url = submit_payload.get("response_url")
        if not status_url or not response_url:
            raise ValueError("fal.ai queue submit response missing status_url/response_url")
        verbose_logger.debug("fal.ai audio polling: rid=%s", submit_payload.get("request_id"))
        return status_url, response_url

    @staticmethod
    def _poll_headers(raw_response: httpx.Response) -> dict[str, str]:
        authorization = raw_response.request.headers.get("Authorization", "")
        return {"Authorization": authorization} if authorization else {}

    def _poll_until_complete_sync(
        self,
        status_url: str,
        headers: dict[str, str],
        client: HTTPHandler,
        timeout_secs: float,
    ) -> None:
        deadline: Final = time.monotonic() + timeout_secs
        while not self._job_completed(client.get(url=status_url, headers=headers), deadline, timeout_secs):
            time.sleep(_POLL_INTERVAL_SECS)

    def _job_completed(self, status_resp: httpx.Response, deadline: float, timeout_secs: float) -> bool:
        self._raise_for_fal_status(status_resp)
        status: Final = (status_resp.json().get("status") or "").upper()
        if status == _TERMINAL_OK:
            return True
        if status in _TERMINAL_FAIL:
            raise RuntimeError(f"fal.ai audio job ended with status={status}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"fal.ai audio job did not complete within {timeout_secs}s")
        return False

    @staticmethod
    def _extract_audio_url(result_payload: dict[str, Any]) -> str:
        error_payload = result_payload.get("error")
        if error_payload:
            raise ValueError(f"fal.ai audio generation failed: {error_payload}")
        audio = result_payload.get("audio")
        if isinstance(audio, dict) and isinstance(audio.get("url"), str):
            return audio["url"]
        audio_url = result_payload.get("audio_url")
        if isinstance(audio_url, str):
            return audio_url
        audio_file = result_payload.get("audio_file")
        if isinstance(audio_file, dict) and isinstance(audio_file.get("url"), str):
            return audio_file["url"]
        raise ValueError(f"fal.ai audio result missing audio url; got keys: {list(result_payload.keys())}")
