from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class FrameMedia:
    first_frames: tuple[str, ...]
    last_frames: tuple[str, ...]
    reference_images: tuple[str, ...]


def resolve_frame_media(
    *,
    image_url: tuple[str, ...],
    end_image_url: tuple[str, ...],
    input_reference: tuple[str, ...],
    image_urls: tuple[str, ...],
    reference_media: tuple[str, ...],
) -> FrameMedia:
    """Split the OpenAI video media slots into SeeGen's frame and reference roles.

    ``input_reference`` names the start frame, the same image ``image_url`` names when both are sent, so it only
    stays a reference image in reference mode (other reference media present and no ``image_url``). A reference
    that repeats a frame URL is dropped because SeeGen refuses frames mixed with reference images.
    """
    input_reference_is_start: Final = not image_url and not image_urls and not reference_media
    first_frames: Final = image_url or (input_reference if input_reference_is_start else ())
    frame_urls: Final = frozenset((*first_frames, *end_image_url))
    candidates: Final = (() if input_reference_is_start else input_reference) + image_urls
    return FrameMedia(
        first_frames=first_frames,
        last_frames=end_image_url,
        reference_images=tuple(url for url in candidates if url not in frame_urls),
    )
