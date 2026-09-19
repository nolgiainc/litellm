from typing import TYPE_CHECKING

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


class FalAIBackgroundRemovalConfig(FalAIBaseConfig):
    """Promptless Bria RMBG 2.0 image background removal."""

    IGNORED_OPENAI_PARAMS: frozenset[str] = frozenset(
        {"prompt", "response_format", "n", "size", "quality", "style", "background"}
    )

    def get_complete_url(
        self,
        api_base: "str | None",
        api_key: "str | None",
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: "bool | None" = None,
    ) -> str:
        base_url = (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        model = model.removeprefix("fal_ai/")
        endpoint = model if model.startswith("fal-ai/") else f"fal-ai/{model}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(self, model: str) -> "list[OpenAIImageGenerationOptionalParams]":
        return ["response_format", "n", "size", "quality", "style", "background"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        return {
            key: value
            for key, value in {**non_default_params, **optional_params}.items()
            if key not in self.IGNORED_OPENAI_PARAMS
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, object],
    ) -> dict[str, object]:
        request = {key: value for key, value in optional_params.items() if key not in self.IGNORED_OPENAI_PARAMS}
        if not request.get("image_url"):
            raise ValueError("Bria background removal requires image_url")
        return request

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        encoding: object,
        api_key: "str | None" = None,
        json_mode: "bool | None" = None,
    ) -> ImageResponse:
        try:
            response_data = TypeAdapter(dict[str, JsonValue]).validate_json(raw_response.content)
        except ValueError as e:
            raise BaseLLMException(
                message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        single = response_data.get("image")
        array = response_data.get("images", [])
        images = (
            [single]
            if isinstance(single, dict)
            else [image for image in array if isinstance(image, dict)]
            if isinstance(array, list)
            else []
        )
        model_response.data = [
            *(model_response.data or []),
            *(
                ImageObject(
                    url=url if isinstance(url := image.get("url"), str) else None,
                    b64_json=b64 if isinstance(b64 := image.get("b64_json"), str) else None,
                )
                for image in images
            ),
        ]
        return model_response
