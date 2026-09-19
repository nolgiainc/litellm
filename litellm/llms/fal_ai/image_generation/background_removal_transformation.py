from collections.abc import Mapping
from itertools import chain
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig, LiteLLMLoggingObj


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
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: "bool | None" = None,
    ) -> str:
        base_url: Final = (api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL).rstrip("/")
        model_id: Final = model.removeprefix("fal_ai/")
        endpoint: Final = model_id if model_id.startswith("fal-ai/") else f"fal-ai/{model_id}"
        return f"{base_url}/{endpoint}"

    def get_supported_openai_params(
        self,
        model: str,
    ) -> "list[OpenAIImageGenerationOptionalParams]":  # mutable-ok: Base image config requires a list return value.
        return [  # mutable-ok: Base config requires a list.
            "response_format",
            "n",
            "size",
            "quality",
            "style",
            "background",
        ]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: The image request pipeline consumes a mutable parameter dictionary.
        return {  # mutable-ok: Build the dictionary required by the image request pipeline in one pass.
            key: value
            for key, value in chain(non_default_params.items(), optional_params.items())
            if key not in self.IGNORED_OPENAI_PARAMS
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: Base image config requires a JSON-serializable request dictionary.
        request: Final = {  # mutable-ok: The HTTP handler serializes this dictionary as the provider request body.
            key: value for key, value in optional_params.items() if key not in self.IGNORED_OPENAI_PARAMS
        }
        if not request.get("image_url"):
            raise ValueError("Bria background removal requires image_url")
        return request

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: object,
        api_key: "str | None" = None,
        json_mode: "bool | None" = None,
    ) -> ImageResponse:
        try:
            response_data: Final = TypeAdapter(dict[str, JsonValue]).validate_json(raw_response.content)
        except ValueError as e:
            raise BaseLLMException(
                message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        single: Final = response_data.get("image")
        array: Final = response_data.get("images", ())
        images: Final = (
            (single,)
            if isinstance(single, dict)
            else tuple(image for image in array if isinstance(image, dict))
            if isinstance(array, list)
            else ()
        )
        data: Final = [  # mutable-ok: ImageResponse.data requires a list, including any existing images.
            *(model_response.data or ()),
            *(
                ImageObject(
                    url=url if isinstance(url := image.get("url"), str) else None,
                    b64_json=b64 if isinstance(b64 := image.get("b64_json"), str) else None,
                )
                for image in images
            ),
        ]
        model_response.data = data  # rebind-ok: The provider contract populates the caller's ImageResponse instance.
        return model_response
