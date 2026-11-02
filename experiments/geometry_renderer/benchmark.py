"""CPU benchmark on the explicitly licensed fixture; never writes photo outputs.

Run from repository root:
  python -m experiments.geometry_renderer.benchmark --iterations 20 --size 512
"""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
from .core import overlay, prepare_face, render
from .detection import decode_image, detect_landmarks


def timed(call):
    start = perf_counter()
    result = call()
    return result, (perf_counter()-start)*1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--size', type=int, choices=(512,960), default=512)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    cv2.setNumThreads(1)
    image = decode_image((Path(__file__).parent/'tests/fixtures/astronaut.png').read_bytes(), '.png')
    if args.size != 512:
        image = np.asarray(Image.fromarray(image).resize((args.size,args.size), Image.Resampling.LANCZOS))
    points, cold_ms = timed(lambda: detect_landmarks(image))
    warm_detect = [timed(lambda: detect_landmarks(image))[1] for _ in range(5)]
    prepared, prepare_ms = timed(lambda: prepare_face(image, points))
    def warp():
        return render(prepared, current_age=20, target_age=80, nose=1, cheeks=1, jowls=1)
    for _ in range(3):
        result = warp()
    warp_times, overlay_times = [], []
    for _ in range(args.iterations):
        result, elapsed = timed(warp)
        warp_times.append(elapsed)
        _, elapsed = timed(lambda: overlay(result.image,result.points,prepared.triangles))
        overlay_times.append(elapsed)
    def stats(values):
        return {"p50_ms": round(float(np.median(values)),2), "p95_ms": round(float(np.percentile(values,95)),2)}
    print(json.dumps({
        'platform': platform.platform(), 'python': platform.python_version(),
        'cpu': platform.processor(), 'logical_cpus_visible': os.cpu_count(),
        'opencv_threads': cv2.getNumThreads(), 'size': list(image.shape[:2]),
        'iterations': args.iterations, 'triangles': len(prepared.triangles),
        'cold_detection_including_import_ms': round(cold_ms,2),
        'warm_detection_5_samples': stats(warm_detect), 'mesh_preparation_ms': round(prepare_ms,2),
        'cached_landmark_warp': stats(warp_times), 'overlay': stats(overlay_times),
        'safety_scale': result.safety_scale,
        'scope': 'Server CPU only; excludes browser transport/display. Public-domain NASA fixture, full strengths, 20→80 slider years. Not a scientific aging model.',
    }, indent=2))

if __name__ == '__main__':
    main()
