"""Offline face parsing and authored, strand-preserving hair graying.

No model/photo downloads or photo writes occur in these rendering functions.
The optional ONNX parser is an independently MIT-licensed BiSeNet project trained
on CelebAMask-HQ; dataset image rights remain separate. This module is a local
academic visual prototype, not a calibrated estimate of graying or biological age.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import os

import cv2
import numpy as np

MODEL_URL = 'https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx'
MODEL_SHA256 = '0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f'
MODEL_FILENAME = 'face_parsing_resnet18.onnx'
HAIR_CLASS = 17
SKIN_CLASSES = (1, 10)  # facial skin and nose; exclude eyes, lips, eyebrows, ears
NECK_CLASS = 14


class HairModelUnavailable(ValueError):
    """Safe message when an explicitly installed parser model is unavailable."""


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Hair layers require an RGB uint8 image')


def model_candidates() -> list[Path]:
    """Only inspect known local locations. Never print environment variable values."""
    paths = []
    override = os.environ.get('AVATAR_FACE_PARSER_MODEL')
    if override:
        paths.append(Path(override).expanduser())
    paths.extend([Path(__file__).parent/'.models'/MODEL_FILENAME,
            Path.home()/'.cache/avatar'/MODEL_FILENAME,
                  Path('/workspace/avatar-runtime/models')/MODEL_FILENAME])
    return paths


def verified_model_path(model_path: Path | None = None) -> Path:
    candidates = [Path(model_path).expanduser()] if model_path is not None else model_candidates()
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise HairModelUnavailable(
            'Local face parser model is missing. Run `python -m '
            'experiments.geometry_renderer.prepare_hair_model` once during setup, '
            'or select its verified local file with AVATAR_FACE_PARSER_MODEL. '
            'Rendering never downloads models.')
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise HairModelUnavailable('Local face parser failed SHA-256 verification; '
                                   'replace it through the explicit model setup command.')
    return path


def head_crop_bounds(image: np.ndarray, landmarks: np.ndarray | None) -> tuple[int,int,int,int]:
    """Fixed source-landmark head crop includes scalp; never rerun face detection."""
    h,w = image.shape[:2]
    if landmarks is None:
        return 0,0,w,h
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 100 or not np.isfinite(points).all():
        raise ValueError('Face parsing needs finite source landmark coordinates')
    low, high = points.min(axis=0), points.max(axis=0)
    width,height = high-low
    if min(width,height) < 10:
        raise ValueError('Face is too small for a head parsing crop')
    side = max(width*2.4, height*2.2)
    center = np.array([(low[0]+high[0])/2, low[1]+height*.30])
    x0,y0 = np.maximum(np.floor(center-side/2),0).astype(int)
    x1,y1 = np.minimum(np.ceil(center+side/2),[w,h]).astype(int)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Head parsing crop is outside the image')
    return int(x0),int(y0),int(x1),int(y1)


def segment_regions(image: np.ndarray, model_path: Path | None = None, *,
                    landmarks: np.ndarray | None = None) -> dict:
    """Return full-size float masks and uint8 labels from one local CPU inference.

    Supply the already detected 468 source landmarks for a fixed head crop. The
    parser expects a face/head image; a whole-body image without landmarks can
    miss or misclassify small hair. Session code should cache this result once
    per selected source and warp its masks with the geometric image if necessary.
    Labels: hair=17, skin=1/nose=10, neck=14; all other classes are protected.
    """
    _validate_rgb(image)
    path = verified_model_path(model_path)
    x0,y0,x1,y1 = head_crop_bounds(image, landmarks)
    head = image[y0:y1,x0:x1]
    resized = cv2.resize(head, (512,512), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255
    # RGB ImageNet normalization matches the model author's reference inference.
    normalized = (resized-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    tensor = np.ascontiguousarray(normalized.transpose(2,0,1)[None])
    try:
        network = cv2.dnn.readNetFromONNX(str(path))
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        network.setInput(tensor)
        logits = network.forward()
    except cv2.error as exc:
        raise HairModelUnavailable('Local face parser could not run with these OpenCV dependencies') from exc
    if logits.shape != (1,19,512,512) or not np.isfinite(logits).all():
        raise HairModelUnavailable('Local face parser returned an unexpected output')
    exponent = np.exp(logits[0]-logits[0].max(axis=0,keepdims=True))
    probabilities = exponent/exponent.sum(axis=0,keepdims=True)
    # A mirrored source-only pass reduces left/right parsing bias. Average
    # semantic probabilities, never expand a hair mask by repainting skin.
    network.setInput(np.ascontiguousarray(tensor[:,:,:,::-1]))
    mirrored = network.forward()[0,:,:,::-1]
    mirrored_exp = np.exp(mirrored-mirrored.max(axis=0,keepdims=True))
    mirrored_prob = mirrored_exp/mirrored_exp.sum(axis=0,keepdims=True)
    # Swap bilateral classes when reflecting the prediction back.
    for left,right in ((2,3),(4,5),(7,8)):
        mirrored_prob[[left,right]] = mirrored_prob[[right,left]]
    probabilities = (probabilities+mirrored_prob)*.5
    labels_small = probabilities.argmax(axis=0).astype(np.uint8)
    crop_size = (x1-x0,y1-y0)
    local_labels = cv2.resize(labels_small,crop_size,interpolation=cv2.INTER_NEAREST)
    labels = np.zeros(image.shape[:2],np.uint8)
    labels[y0:y1,x0:x1] = local_labels
    masks = {}
    for name,classes in [('hair',(HAIR_CLASS,)),('skin',SKIN_CLASSES),('neck',(NECK_CLASS,)),('ears',(7,8))]:
        # Conservative winner gate prevents low-confidence hair probability
        # leaking onto an explicitly classified skin/background pixel.
        confidence = probabilities[list(classes)].sum(axis=0)
        confidence *= np.isin(labels_small,classes)
        local_mask = cv2.resize(confidence,crop_size,interpolation=cv2.INTER_LINEAR)
        local_mask *= np.isin(local_labels,classes)
        mask = np.zeros(image.shape[:2],np.float32)
        mask[y0:y1,x0:x1] = np.clip(local_mask,0,1)
        masks[name] = mask
    warnings = []
    if landmarks is None:
        warnings.append('Parser expects a head/face crop; small faces in full portraits may be missed.')
    if not np.any(masks['hair'] > .5):
        warnings.append('No confident hair region was found; hair graying will make no visible change.')
    warnings.append('Learned masks can be wrong or miss long hair; inspect the layer mask before interpreting the result.')
    return {**masks,'labels':labels,'backend':'BiSeNet ResNet18 ONNX / OpenCV CPU',
            'warnings':warnings,'head_crop':(x0,y0,x1,y1),'model_sha256':MODEL_SHA256}


def segment_hair(image: np.ndarray, model_path: Path | None = None, *,
                 landmarks: np.ndarray | None = None) -> np.ndarray:
    """Convenience hair-only adapter; use segment_regions once if skin is needed."""
    return segment_regions(image,model_path,landmarks=landmarks)['hair']


def conservative_hair_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """UNCERTAIN nonsemantic fallback, never silently substituted for a parser.

    Detects only darker scalp colors above/outside the face hull in an authored
    head ellipse. May miss blonde/gray/long hair and confuse dark backgrounds.
    The caller must label this as uncertain and offer a disable/inspect control.
    """
    _validate_rgb(image)
    points = np.asarray(landmarks,np.float64)
    head_crop_bounds(image,points)  # validate coordinates
    h,w = image.shape[:2]
    low,high = points.min(axis=0),points.max(axis=0)
    fw,fh = high-low
    x,y = np.meshgrid(np.arange(w),np.arange(h))
    cx=(low[0]+high[0])/2;cy=low[1]-.05*fh
    scalp=((x-cx)/(.7*fw))**2+((y-cy)/(.48*fh))**2 < 1
    face=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(face,cv2.convexHull(np.rint(points).astype(np.int32)),1)
    face=cv2.dilate(face,np.ones((5,5),np.uint8))
    luminance=image.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32)
    # Deliberately exclude bright pixels instead of asserting that they are hair.
    candidate=scalp & (face==0) & (luminance<135)
    mask=candidate.astype(np.float32)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    return mask


def gray_hair(image: np.ndarray, mask: np.ndarray, strength: float, *, seed: int = 2026) -> tuple[np.ndarray,dict]:
    """Conservative tonal graying while retaining source hair texture.

    Strength is an illustration control, not a clinical fraction of gray hair.
    Chroma moves toward neutral with bounded luminance lift and no added noise.
    Bright outliers and soft boundary pixels are protected from whitening.
    Pixels outside the supplied confident mask are bit-for-bit protected.
    """
    _validate_rgb(image)
    mask=np.asarray(mask,np.float32)
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or (mask<0).any() or (mask>1).any():
        raise ValueError('Hair mask must match the image and contain finite values between 0 and 1')
    if not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Hair graying strength must be finite and between 0 and 1')
    if not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError('Hair texture seed must be a nonnegative integer')
    support=mask>.5
    metrics={'hair_pixels':int(support.sum()),'strength':float(strength),'applied':False,'mean_change':0.0}
    if strength==0 or not support.any():
        return image.copy(),metrics
    rgb=image.astype(np.float32)
    lum=rgb@np.array([.2126,.7152,.0722],np.float32)
    # Never select silver patches from source highlights: bright background or
    # forehead leakage otherwise becomes a white rim/strip through the fringe.
    distance=cv2.distanceTransform(support.astype(np.uint8),cv2.DIST_L2,3)
    feather=max(3.,min(10.,min(image.shape[:2])*.012))
    core=support & (distance>feather+2) & (mask>.5)
    if core.sum()<32:
        metrics.update(mode='shadow_preserving_ash_gray',reason='no_reliable_hair_interior')
        return image.copy(),metrics
    low,high=np.percentile(lum[core],[15,50])
    cutoff=float(high)+max(18.,float(high-low)*2.)
    # Bright outliers relative to the hair interior are protected, including
    # skin/wall pixels confidently misclassified by the semantic parser.
    brightness_gate=np.clip((cutoff+18-lum)/18,0,1)
    confidence=np.clip((mask-.5)/.10,0,1)
    edge=np.clip((distance-1)/feather,0,1)
    alpha=confidence*edge*brightness_gate*.72*float(strength)
    # A smooth saturating lift makes dark hair visibly ash-gray without
    # selecting bright patches. Black shadows stay black; no new strands/noise.
    silver=np.clip(lum+32*(-np.expm1(-lum/25)),0,255)
    neutral=np.repeat(silver[...,None],3,axis=2)
    result=np.rint(np.clip(rgb*(1-alpha[...,None])+neutral*alpha[...,None],0,255)).astype(np.uint8)
    result[~support]=image[~support]
    metrics.update(mode='shadow_preserving_ash_gray',
        retained_source_fraction=float(np.mean(np.all(result[support]==image[support],axis=1))),
        protected_bright_pixels=int(np.sum(support & (brightness_gate==0))),
        feather_pixels=float(feather),max_luminance_lift=24.,
        selection_note='Conservative source-tone neutralization; not follicle synthesis or a measured gray-hair percentage.')
    metrics.update(applied=bool(np.any(result!=image)),
        mean_change=float(np.abs(result.astype(np.float32)-rgb)[support].mean()),
        mean_luminance_before=float(lum[support].mean()),
        mean_luminance_after=float((result.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32))[support].mean()))
    return result,metrics
