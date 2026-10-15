"""Explicit integration point for AVATAR's image-generation endpoint.

Call store_pair() server-side only after generation succeeds and after IRB approval.
Never invoke this from a browser, and never include Box file IDs in public URLs.
"""
from __future__ import annotations

from dataclasses import dataclass

from avatar.box_storage import BoxUploadResult, new_participant_id, upload_study_image, delete_study_image


@dataclass(frozen=True)
class StoredImagePair:
    participant_id: str
    original: BoxUploadResult
    aged: BoxUploadResult


def store_pair(*, original: bytes, aged: bytes,
               original_content_type: str, aged_content_type: str,
               token_provider, participant_id: str | None = None) -> StoredImagePair:
    """Upload both images with one random participant ID.

    Important: Box uploads are not atomic. If the second upload fails, the first
    remains stored. The calling backend must handle reconciliation/cleanup.
    Do not report a fully stored pair unless both uploads succeed.
    """
    pid = participant_id or new_participant_id()
    first = upload_study_image(original, participant_id=pid,
                               image_kind="original", content_type=original_content_type,
                               token_provider=token_provider)
    try:
        second = upload_study_image(aged, participant_id=pid,
                                    image_kind="aged", content_type=aged_content_type,
                                    token_provider=token_provider)
    except Exception:
        try:
            delete_study_image(first.file_id, token_provider=token_provider)
        except Exception as cleanup_exc:
            raise RuntimeError(
                f"Box pair incomplete; cleanup failed for participant {pid}, "
                f"original file {first.file_id}. Manual reconciliation required."
            ) from cleanup_exc
        raise
    return StoredImagePair(participant_id=pid, original=first, aged=second)
