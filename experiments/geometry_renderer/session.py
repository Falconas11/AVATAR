"""Session-local memory only; no Streamlit disk/global photo caches."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from .core import PreparedFace, prepare_face
from .detection import decode_image, detect_landmarks


@dataclass
class DemoSession:
    prepared: PreparedFace | None = None
    _key: tuple[str, str] | None = None
    detection_ms: float = 0
    preparation_ms: float = 0
    detection_count: int = 0

    def clear(self) -> None:
        self.prepared = None
        self._key = None
        self.detection_ms = self.preparation_ms = 0
        self.detection_count = 0

    def load(self, data: bytes, suffix: str, *, detector=detect_landmarks) -> PreparedFace:
        key = (sha256(data).hexdigest(), suffix.lower())
        if self._key == key and self.prepared is not None:
            return self.prepared
        self.clear()
        image = decode_image(data, suffix)
        start = perf_counter()
        landmarks = detector(image)
        self.detection_ms = (perf_counter()-start)*1000
        start = perf_counter()
        self.prepared = prepare_face(image, landmarks)
        self.preparation_ms = (perf_counter()-start)*1000
        self.detection_count += 1
        self._key = key
        return self.prepared
