"""Authored, unitless displacement hypotheses and continuous inverse triangle maps.

MediaPipe's 468-index topology identifies regions; it is NOT an aging model.
All fractions below are conservative visual design choices, not measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

# MediaPipe Face Mesh indices (unrefined 468-landmark topology).
# Paired anchors limit asymmetry; the compact field decays to zero at its radius.
REGIONS = {
    "nose": (1, 2, 4),                 # tip / inferior tip / bridge-to-tip
    "cheeks": (50, 101, 205, 280, 330, 425),  # left / right midface
    "jowls": (172, 136, 150, 397, 365, 379),  # paired lower facial contour
}
# (vertical shift / face height at full strength, radius / face width)
FIELDS = {"nose": (0.012, 0.18), "cheeks": (0.018, 0.28), "jowls": (0.022, 0.25)}


@dataclass(frozen=True)
class PreparedFace:
    image: np.ndarray
    points: np.ndarray
    triangles: np.ndarray
    face_count: int


@dataclass(frozen=True)
class WarpResult:
    image: np.ndarray
    points: np.ndarray
    safety_scale: float


def signed_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = points[triangles].transpose(1, 0, 2)
    return ((b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) -
            (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])) / 2


def boundary_points(width: int, height: int) -> np.ndarray:
    """Corners and evenly spaced edge anchors, pinned during all deformations."""
    x = np.linspace(0, width-1, 9)
    y = np.linspace(0, height-1, 9)[1:-1]
    return np.array([(v, 0) for v in x] + [(v, height-1) for v in x] +
                    [(0, v) for v in y] + [(width-1, v) for v in y], dtype=np.float64)


def prepare_face(image: np.ndarray, landmarks: np.ndarray) -> PreparedFace:
    """Precompute once per image. Landmarks are pixel coordinates, never normalized."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image")
    h, w = image.shape[:2]
    face = np.asarray(landmarks, dtype=np.float64)
    if face.shape != (468, 2) or not np.isfinite(face).all():
        raise ValueError("Exactly 468 finite face landmarks are required")
    if w < 64 or h < 64 or (face < 1).any() or (face > [w-2, h-2]).any():
        raise ValueError("Use a fully visible face away from the image edges")
    if np.ptp(face[:, 0]) < 20 or np.ptp(face[:, 1]) < 20:
        raise ValueError("Face is too small for a usable mesh")
    points = np.vstack((face, boundary_points(w, h)))
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Duplicate landmarks; try a clearer, front-facing portrait")
    try:
        triangles = Delaunay(points).simplices.copy()
    except QhullError as exc:
        raise ValueError("Could not form a usable face mesh") from exc
    areas = signed_areas(points, triangles)
    triangles[areas < 0] = triangles[areas < 0][:, [0, 2, 1]]
    if (signed_areas(points, triangles) < 1e-4).any():
        raise ValueError("Degenerate face mesh; try a clearer portrait")
    return PreparedFace(image.copy(), points, triangles, len(face))


def displacement_field(points: np.ndarray, face_count: int, *, current_age: int,
                       target_age: int, nose: float, cheeks: float, jowls: float) -> np.ndarray:
    """Smooth compact radial fields. Age fraction is illustrative, not calibrated.

    Equal current/target age or zero strengths is exact identity. Strengths are
    [0,1]. Full field scale is reached after 60 slider years, without implying
    an empirical rate. Boundary points have exactly zero displacement.
    """
    if (type(current_age) is not int or type(target_age) is not int or
            not 18 <= current_age <= target_age <= 100):
        raise ValueError("Use integer ages 18–100 with target at least current age")
    strengths = {"nose": nose, "cheeks": cheeks, "jowls": jowls}
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in strengths.values()):
        raise ValueError("Region strengths must be finite values between 0 and 1")
    face = points[:face_count]
    width, height = np.ptp(face, axis=0)
    delta = np.zeros_like(points, dtype=np.float64)
    age_fraction = min((target_age-current_age)/60, 1.0)
    for region, indices in REGIONS.items():
        shift, radius = FIELDS[region]
        distance = np.linalg.norm(face[:, None, :] - face[list(indices)][None, :, :], axis=2)
        r = np.clip(distance/(width*radius), 0, 1)
        # Smooth bounded union of Wendland C2 kernels. Unlike max(), this
        # avoids gradient cusps where neighboring anchor fields cross.
        kernels = (1-r)**4 * (4*r+1)
        falloff = 1-np.prod(1-kernels, axis=1)
        delta[:face_count, 1] += falloff * height * shift * strengths[region] * age_fraction
    return delta


def safe_destination(prepared: PreparedFace, delta: np.ndarray) -> tuple[np.ndarray, float]:
    """Backtrack the WHOLE field until every triangle preserves area/orientation.

    Positive Jacobians plus the fixed convex boundary preserve the mesh mapping.
    A 10% source-area floor also prevents almost-flat triangles.
    """
    points, triangles = prepared.points, prepared.triangles
    if delta.shape != points.shape or not np.isfinite(delta).all():
        raise ValueError("Displacement must match the finite mesh coordinates")
    delta = delta.copy()
    delta[prepared.face_count:] = 0
    source_area = signed_areas(points, triangles)
    h, w = prepared.image.shape[:2]
    scale = 1.0
    for _ in range(24):
        candidate = points + delta*scale
        if ((candidate >= 0).all() and (candidate <= [w-1, h-1]).all() and
                (signed_areas(candidate, triangles) >= np.maximum(source_area*.1, 1e-5)).all()):
            return candidate, scale
        scale *= .5
    return points.copy(), 0.0


def inverse_maps(source: np.ndarray, destination: np.ndarray, triangles: np.ndarray,
                 width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize shared continuous inverse affine maps, then sample once.

    Barycentric interpolation agrees on shared triangle edges, avoiding independent
    per-triangle image masks/blends and their holes/double-composited seams.
    """
    map_x = np.full((height, width), -1, dtype=np.float32)
    map_y = np.full_like(map_x, -1)
    for indices in triangles:
        dst, src = destination[indices], source[indices]
        low = np.maximum(np.floor(dst.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(dst.max(axis=0)).astype(int), [width-1, height-1])
        # Broadcast 1-D coordinates and use the analytic 2x2 inverse. Avoid
        # allocating stacked per-pixel vectors or inverting 966 small matrices.
        y, x = np.ogrid[low[1]:high[1]+1, low[0]:high[0]+1]
        first, second = dst[1]-dst[0], dst[2]-dst[0]
        determinant = first[0]*second[1]-first[1]*second[0]
        if determinant <= 0:
            raise ValueError("Cannot rasterize an inverted or degenerate triangle")
        u = (second[1]*(x-dst[0,0])-second[0]*(y-dst[0,1]))/determinant
        v = (-first[1]*(x-dst[0,0])+first[0]*(y-dst[0,1]))/determinant
        inside = (u >= -1e-7) & (v >= -1e-7) & (u+v <= 1+1e-7)
        region_x = map_x[low[1]:high[1]+1, low[0]:high[0]+1]
        region_y = map_y[low[1]:high[1]+1, low[0]:high[0]+1]
        region_x[inside] = (src[0,0]+u*(src[1,0]-src[0,0])+v*(src[2,0]-src[0,0]))[inside]
        region_y[inside] = (src[0,1]+u*(src[1,1]-src[0,1])+v*(src[2,1]-src[0,1]))[inside]
    if (map_x < -.001).any() or (map_y < -.001).any():
        raise ValueError("Mesh did not cover the image; refusing an incomplete warp")
    return np.clip(map_x, 0, width-1), np.clip(map_y, 0, height-1)


def render(prepared: PreparedFace, *, current_age: int, target_age: int,
           nose: float = .5, cheeks: float = .5, jowls: float = .5) -> WarpResult:
    delta = displacement_field(prepared.points, prepared.face_count, current_age=current_age,
                               target_age=target_age, nose=nose, cheeks=cheeks, jowls=jowls)
    destination, scale = safe_destination(prepared, delta)
    if np.array_equal(destination, prepared.points):
        return WarpResult(prepared.image.copy(), destination, scale)
    h, w = prepared.image.shape[:2]
    x, y = inverse_maps(prepared.points, destination, prepared.triangles, w, h)
    warped = cv2.remap(prepared.image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return WarpResult(warped, destination, scale)


def overlay(image: np.ndarray, points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    result = image.copy()
    cv2.polylines(result, np.rint(points[triangles]).astype(np.int32), True, (30, 210, 180), 1, cv2.LINE_AA)
    for x, y in np.rint(points[:468]).astype(int):
        cv2.circle(result, (x, y), 1, (255, 180, 30), -1)
    return result
