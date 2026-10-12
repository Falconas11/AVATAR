"""Optional server-side Box uploader for AVATAR pilot research images.

This module does not authorize research collection or provision Box credentials.
Use only a PI-approved, restricted-access UIUC Box folder and approved authentication.
Never send Box tokens to the browser or commit them to source control.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import urllib.error
import urllib.request
from dataclasses import dataclass

MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
PARTICIPANT_ID_PATTERN = re.compile(r"^P-[0-9a-f]{32}$")
BOX_UPLOAD_URL = "https://upload.box.com/api/2.0/files/content"


class BoxStorageError(RuntimeError):
    """An image could not be stored in Box."""


@dataclass(frozen=True)
class BoxUploadResult:
    file_id: str
    participant_id: str
    image_kind: str


def new_participant_id() -> str:
    """Random study code; do not use a name, email address, or sequential ID."""
    return "P-" + secrets.token_hex(16)


def _validated_name(participant_id: str, image_kind: str, content_type: str) -> str:
    if not PARTICIPANT_ID_PATTERN.fullmatch(participant_id):
        raise ValueError("Invalid participant ID")
    if image_kind not in ("original", "aged"):
        raise ValueError("image_kind must be 'original' or 'aged'")
    if content_type not in ALLOWED_TYPES:
        raise ValueError("Unsupported image content type")
    return f"{participant_id}_{image_kind}.{ALLOWED_TYPES[content_type]}"


def upload_study_image(
    image: bytes,
    *,
    participant_id: str,
    image_kind: str,
    content_type: str,
    access_token: str | None = None,
    folder_id: str | None = None,
    timeout: int = 30,
) -> BoxUploadResult:
    """Upload an image to a preconfigured Box folder; never return a public URL.

    A short-lived access token may be supplied by an approved server-side token
    provider. BOX_ACCESS_TOKEN is suitable for a development smoke test only:
    production must manage token renewal securely.
    """
    filename = _validated_name(participant_id, image_kind, content_type)
    if not image or len(image) > MAX_IMAGE_BYTES:
        raise ValueError("Image must be between 1 byte and 10 MiB")
    if not isinstance(image, bytes):
        raise TypeError("image must be bytes")

    token = access_token or os.environ.get("BOX_ACCESS_TOKEN")
    parent_id = folder_id or os.environ.get("BOX_STUDY_FOLDER_ID")
    if not token or not parent_id or not parent_id.isdecimal():
        raise BoxStorageError("Box storage is not configured")

    boundary = "avatar-" + secrets.token_hex(16)
    attributes = json.dumps(
        {"name": filename, "parent": {"id": parent_id}},
        separators=(",", ":"),
    ).encode("utf-8")
    b = boundary.encode("ascii")
    body = (
        b"--" + b + b"\r\n"
        + b'Content-Disposition: form-data; name="attributes"\r\n'
        + b"Content-Type: application/json\r\n\r\n"
        + attributes + b"\r\n"
        + b"--" + b + b"\r\n"
        + f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
        + f"Content-Type: {content_type}\r\n\r\n".encode()
        + image + b"\r\n"
        + b"--" + b + b"--\r\n"
    )
    request = urllib.request.Request(
        BOX_UPLOAD_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
        file_id = str(payload["entries"][0]["id"])
        if not file_id.isdecimal():
            raise ValueError("Unexpected Box file ID")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            ValueError, KeyError, IndexError, TypeError) as exc:
        # Do not log response bodies or tokens: they may contain sensitive data.
        raise BoxStorageError("Box image upload failed; storage is not confirmed") from exc
    return BoxUploadResult(file_id=file_id, participant_id=participant_id, image_kind=image_kind)
