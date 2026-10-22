"""Aligned inspection views; these functions do not add any deformation."""
from __future__ import annotations

import numpy as np
from .core import PreparedFace, REGIONS


def face_crop_bounds(prepared: PreparedFace) -> tuple[int, int, int, int]:
    """Fixed source-based crop reused at every age; no independent auto-zoom."""
    face = prepared.points[:prepared.face_count]
    low, high = face.min(axis=0), face.max(axis=0)
    padding = (high-low)*.35
    h, w = prepared.image.shape[:2]
    x0, y0 = np.maximum(np.floor(low-padding), 0).astype(int)
    x1, y1 = np.minimum(np.ceil(high+padding)+1, [w,h]).astype(int)
    return int(x0), int(y0), int(x1), int(y1)


def crop_view(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    x0,y0,x1,y1 = bounds
    return image[y0:y1,x0:x1].copy()


def wipe_view(original: np.ndarray, warped: np.ndarray, percent: int) -> np.ndarray:
    """Original on the left, warped on the right, with matching coordinates."""
    if original.shape != warped.shape or not 0 <= percent <= 100:
        raise ValueError('Wipe requires matching images and a percent from 0 to 100')
    split = round(original.shape[1]*percent/100)
    result = warped.copy()
    result[:, :split] = original[:, :split]
    if 0 < split < original.shape[1]:
        result[:, split] = (255,190,40)
    return result


def difference_view(original: np.ndarray, warped: np.ndarray) -> np.ndarray:
    """Amplified RGB absolute pixel difference; diagnostic, never an aged image."""
    return np.clip(np.abs(original.astype(np.int16)-warped.astype(np.int16))*4, 0, 255).astype(np.uint8)


def applied_shifts(prepared: PreparedFace, destination: np.ndarray) -> dict[str, float]:
    """Measured after safety limiting, in resized input pixels, not mm."""
    distances = np.linalg.norm(destination-prepared.points, axis=1)
    return {**{name: float(distances[list(indices)].mean()) for name,indices in REGIONS.items()},
            'maximum': float(distances[:prepared.face_count].max())}
