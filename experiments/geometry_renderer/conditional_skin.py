"""Optional local age-conditioned learned appearance; geometry stays mesh-driven.

The model is Tim Roelofs' independently MIT-licensed FRAN-style reimplementation,
not Disney's proprietary official model. Source/target age are neural conditions,
not calibrated predictions. AVATAR graph references do not explain this network.
Runtime uses verified local ONNX + OpenCV CPU: no torch, uploads, or downloads.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import os
from time import perf_counter
import cv2
import numpy as np
from .hair_layers import head_crop_bounds

MODEL_FILENAME='fran_reaging.onnx'
MODEL_SHA256='5941608e21f5d218ec52862938898efb276b63048aa8586372ff046ef865e22b'
MODEL_BYTES=124230737


class ConditionalModelUnavailable(ValueError):
    """Safe local-model setup error; never contains an input-photo filename."""


def verified_model_path(model_path: Path | None=None) -> Path:
    if model_path is None:
        override=os.environ.get('AVATAR_CONDITIONAL_AGING_MODEL')
        candidates=([Path(override).expanduser()] if override else [])+[
            Path(__file__).parent/'.models'/MODEL_FILENAME,
            Path.home()/'.cache/avatar'/MODEL_FILENAME,
            Path('/workspace/avatar-runtime/models')/MODEL_FILENAME]
    else:
        candidates=[Path(model_path).expanduser()]
    path=next((candidate for candidate in candidates if candidate.is_file()),None)
    if path is None:
        raise ConditionalModelUnavailable('Optional local skin model is missing. Run `python -m '
            'experiments.geometry_renderer.prepare_conditional_model` during setup. '
            'Mesh-only rendering does not require this model; runtime never downloads it.')
    if path.stat().st_size!=MODEL_BYTES or sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:
        raise ConditionalModelUnavailable('Optional local skin model failed size/SHA-256 verification; '
            'replace it using the explicit model setup command.')
    return path


def _windowed_residual(network, rgb: np.ndarray, current_age: int, target_age: int) -> tuple[np.ndarray,int]:
    """512-window overlap-add, with positive raised-cosine coverage everywhere.

    A512 canvas uses one window;768 uses four and1024 uses nine overlapping windows. The quality-first ultra mode uses nine windows on a1024 canvas.
    """
    size=rgb.shape[0]
    if rgb.shape!=(size,size,3) or size not in (512,768,1024):
        raise ValueError('Conditional work canvas must be512 or768 square RGB')
    summed=np.zeros_like(rgb,dtype=np.float32)
    covered=np.zeros((size,size),np.float32)
    weights=np.maximum(np.outer(np.hanning(512),np.hanning(512)),1e-3).astype(np.float32)
    count=0
    for y in range(0,size-511,256):
        for x in range(0,size-511,256):
            tile=rgb[y:y+512,x:x+512]
            tensor=np.concatenate([tile.transpose(2,0,1),
                np.full((1,512,512),current_age/100,np.float32),
                np.full((1,512,512),target_age/100,np.float32)])[None]
            network.setInput(np.ascontiguousarray(tensor))
            delta=network.forward()
            if delta.shape!=(1,3,512,512) or not np.isfinite(delta).all():
                raise ConditionalModelUnavailable('Optional skin model returned unexpected pixels')
            summed[y:y+512,x:x+512]+=delta[0].transpose(1,2,0)*weights[...,None]
            covered[y:y+512,x:x+512]+=weights
            count+=1
    if (covered<=0).any():
        raise ConditionalModelUnavailable('Conditional detail windows did not cover the crop')
    return summed/covered[...,None],count


def generate_aged_skin(image: np.ndarray,landmarks: np.ndarray,*,current_age: int,
                       target_age: int,model_path: Path | None=None,
                       quality: str='fast') -> tuple[np.ndarray,dict]:
    """Return an optional age-conditioned proposal, full-size RGB uint8.

    Callers MUST mask the proposal to intended skin regions: raw residuals may
    alter eyes, hair, background, identity or lighting inside the head crop.
    Outside that fixed crop pixels are exact. Cache per image/current/target;
    changes to comparison controls should never cause another neural inference.
    Equal ages are short-circuited to exact identity without model loading.
    """
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:
        raise ValueError('Conditional skin requires an RGB uint8 image')
    if type(current_age) is not int or type(target_age) is not int or not 18<=current_age<=target_age<=100:
        raise ValueError('Use integer ages18–100, target at least current age')
    if quality not in {'fast','detail','ultra'}:
        raise ValueError('Conditional skin quality must be fast or detail')
    metrics={'quality':quality,'forward_count':0,'working_size':{'fast':512,'detail':768,'ultra':1024}[quality],
        'backend':'Optional FRAN-style age-conditioned skin / OpenCV CPU',
        'source_age_condition':current_age,'target_age_condition':target_age,
        'applied':False,'inference_ms':0.0,'mean_head_change':0.0,
        'model_sha256':MODEL_SHA256,'clinically_calibrated':False,
        'disclaimer':'Optional learned age appearance, not a calibrated individual prediction or knowledge-graph mechanism explanation. Geometry is controlled separately by the mesh.'}
    if current_age==target_age:
        return image.copy(),metrics
    path=verified_model_path(model_path)
    x0,y0,x1,y1=head_crop_bounds(image,landmarks)
    source=image[y0:y1,x0:x1]
    size={'fast':512,'detail':768,'ultra':1024}[quality]
    rgb=cv2.resize(source,(size,size),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255
    start=perf_counter()
    try:
        network=cv2.dnn.readNetFromONNX(str(path))
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        delta,forward_count=_windowed_residual(network,rgb,current_age,target_age)
    except cv2.error as exc:
        raise ConditionalModelUnavailable('The verified optional skin model could not run with OpenCV CPU') from exc
    # Residual inference preserves the original sampling rather than replacing
    # the whole face by a GAN reconstruction. It can still hallucinate details.
    residual=cv2.resize(delta,(x1-x0,y1-y0),interpolation=cv2.INTER_CUBIC)
    proposed=np.clip(np.rint(source.astype(np.float32)+residual*255),0,255).astype(np.uint8)
    output=image.copy();output[y0:y1,x0:x1]=proposed
    metrics.update(applied=bool(np.any(proposed!=source)),forward_count=forward_count,inference_ms=(perf_counter()-start)*1000,
        mean_head_change=float(np.abs(proposed.astype(np.float32)-source.astype(np.float32)).mean()),
        head_crop=(x0,y0,x1,y1))
    return output,metrics
