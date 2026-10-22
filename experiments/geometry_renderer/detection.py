"""Bounded image decoding and offline, CPU MediaPipe Face Mesh detection."""
from __future__ import annotations

from io import BytesIO
import warnings
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
MAX_SIDE = 960


class PortraitError(ValueError):
    """Safe user-facing validation message; contains no photo data or paths."""


def decode_image(data: bytes, suffix: str) -> np.ndarray:
    if suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise PortraitError("Choose a JPEG or PNG file")
    if not data or len(data) > MAX_BYTES:
        raise PortraitError("Image must be nonempty and at most 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                expected = 'PNG' if suffix.lower() == '.png' else 'JPEG'
                if image.format != expected:
                    raise PortraitError("File contents do not match the JPEG/PNG extension")
                if image.width * image.height > MAX_PIXELS or min(image.size) < 64:
                    raise PortraitError("Use an image at least 64 pixels per side and at most 16 megapixels")
                if getattr(image, 'n_frames', 1) != 1:
                    raise PortraitError("Use a still JPEG or PNG, not an animated image")
                image = ImageOps.exif_transpose(image).convert('RGB')
                image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
                return np.asarray(image).copy()
    except PortraitError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PortraitError("Could not decode a valid, bounded JPEG/PNG image") from exc


def select_face(faces, width: int, height: int) -> np.ndarray:
    if not faces:
        raise PortraitError("No usable face detected. Try a clear, front-facing portrait")
    if len(faces) != 1:
        raise PortraitError("Multiple faces detected. Select a portrait containing exactly one face")
    landmarks = faces[0].landmark
    if len(landmarks) != 468:
        raise PortraitError("Incomplete face mesh. Try a clearer portrait")
    points = np.array([(p.x*(width-1), p.y*(height-1)) for p in landmarks], dtype=np.float64)
    if (not np.isfinite(points).all() or (points < 1).any() or
            (points > [width-2, height-2]).any() or
            np.ptp(points[:, 0]) < 20 or np.ptp(points[:, 1]) < 20):
        raise PortraitError("Face is clipped or too small. Keep the entire face inside the image")
    return points


def detect_landmarks(image: np.ndarray) -> np.ndarray:
    # Pinned MediaPipe 0.10.21 ships these TFLite models in its wheel. No download
    # or network client is used at runtime. Request 2 faces so multiple faces fail.
    import mediapipe as mp
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
            refine_landmarks=False, min_detection_confidence=.5) as mesh:
        result = mesh.process(image)
    return select_face(result.multi_face_landmarks, image.shape[1], image.shape[0])
