"""Session-local preparation, semantic masks and one render result, in memory."""
from __future__ import annotations
from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter
import cv2
import numpy as np
from .core import PreparedFace, prepare_face
from .detection import decode_image, detect_landmarks

TARGET_FACE_HEIGHT = 300
MAX_WORKING_SIDE = 1536


@dataclass
class DemoSession:
    prepared: PreparedFace | None = None
    _key: tuple[str,str] | None = None
    regions: dict | None = None
    aged_proposal: np.ndarray | None = field(default=None,repr=False)
    proposal_metrics: dict | None = None
    proposal_ms: float = 0
    proposal_count: int = 0
    _proposal_key: tuple[int,int,str] | None = None
    detection_ms: float = 0
    preparation_ms: float = 0
    segmentation_ms: float = 0
    detection_count: int = 0
    segmentation_count: int = 0
    render_count: int = 0
    working_scale: float = 1
    _render_key: tuple | None = None
    _render_result: object | None = field(default=None,repr=False)

    def clear(self) -> None:
        self.prepared = self.regions = self._key = self._render_key = self._render_result = self.aged_proposal = None
        self.proposal_metrics = self._proposal_key = None
        self.proposal_ms = 0
        self.proposal_count = 0
        self.detection_ms = self.preparation_ms = self.segmentation_ms = 0
        self.detection_count = self.segmentation_count = self.render_count = 0
        self.working_scale = 1

    def load(self, data: bytes, suffix: str, *, detector=detect_landmarks) -> PreparedFace:
        key = (sha256(data).hexdigest(),suffix.lower())
        if self._key == key and self.prepared is not None:
            return self.prepared
        self.clear()
        image = decode_image(data,suffix)
        start = perf_counter()
        landmarks = detector(image)
        self.detection_ms = (perf_counter()-start)*1000
        start = perf_counter()
        face_height = float(np.ptp(landmarks[:,1]))
        # Paint new detail at a useful working resolution rather than squeezing
        # wrinkles into the sample's ~100-pixel face. Upsampling adds no real
        # source detail and is bounded; it is an explicit rasterization aid.
        scale = max(1.,min(TARGET_FACE_HEIGHT/max(face_height,1),MAX_WORKING_SIDE/max(image.shape[:2])))
        if scale > 1.01:
            h,w = image.shape[:2]
            new_w,new_h = int(round(w*scale)),int(round(h*scale))
            image = cv2.resize(image,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
            landmarks = landmarks*np.array([(new_w-1)/(w-1),(new_h-1)/(h-1)])
        self.working_scale = scale
        self.prepared = prepare_face(image,landmarks)
        self.preparation_ms = (perf_counter()-start)*1000
        self.detection_count += 1
        self._key = key
        return self.prepared

    def ensure_regions(self, *, segmenter=None) -> dict:
        if self.prepared is None:
            raise ValueError('Select a portrait before preparing semantic masks')
        if self.regions is None:
            if segmenter is None:
                from .hair_layers import segment_regions
                segmenter = segment_regions
            start = perf_counter()
            self.regions = segmenter(self.prepared.image,landmarks=self.prepared.points[:468])
            self.segmentation_ms = (perf_counter()-start)*1000
            self.segmentation_count += 1
        return self.regions

    def ensure_aged_proposal(self, current_age: int = 30, target_age: int = 75, *, quality: str = 'detail', generator=None) -> np.ndarray:
        if self.prepared is None:
            raise ValueError('Select a portrait before preparing aged appearance')
        key = (current_age,target_age,quality)
        if self.aged_proposal is None or self._proposal_key != key:
            if generator is None:
                from .conditional_skin import generate_aged_skin
                generator = generate_aged_skin
            start = perf_counter()
            proposal,metrics = generator(self.prepared.image,self.prepared.points[:468],
                current_age=current_age,target_age=target_age,quality=quality)
            if proposal.shape != self.prepared.image.shape or proposal.dtype != np.uint8:
                raise ValueError('Aged appearance proposal must match the prepared RGB image')
            self.aged_proposal = proposal
            self.proposal_metrics = metrics
            self._proposal_key = key
            self.proposal_ms = (perf_counter()-start)*1000
            self.proposal_count += 1
            self._render_key = self._render_result = None
        return self.aged_proposal

    def render_age(self, current_age: int, target_age: int, *, lifestyle: dict | None=None,
                   geometry_strengths: dict | None=None, skin_strength: float=1,
                   hair_strength: float=1, enabled_layers=('geometry','skin','hair'),renderer=None):
        if self.prepared is None:
            raise ValueError('Select a portrait before rendering')
        key = (current_age,target_age,tuple(sorted((lifestyle or {}).items())),
            tuple(sorted((geometry_strengths or {}).items())),skin_strength,hair_strength,tuple(sorted(enabled_layers)))
        if key == self._render_key and self._render_result is not None:
            return self._render_result
        if renderer is None:
            from .pipeline import render_local_aging
            renderer = render_local_aging
        self._render_result = renderer(self.prepared,self.regions,current_age,target_age,lifestyle=lifestyle,
            learned_proposal=self.aged_proposal,
            geometry_strengths=geometry_strengths,skin_strength=skin_strength,
            hair_strength=hair_strength,enabled_layers=enabled_layers)
        self._render_key = key
        self.render_count += 1
        return self._render_result
