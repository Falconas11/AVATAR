"""Reproducible complete local photo-processing benchmark and 20-second gate.

Uses only the public-domain NASA fixture, never a participant photo. Model setup
must be completed first. Writes JSON measurements only, never photo outputs.
"""
from __future__ import annotations
from time import perf_counter
PROCESS_START=perf_counter()
import argparse
import json
import os
from pathlib import Path
import platform


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality',choices=('fast','detail','ultra'),default='detail')
    parser.add_argument('--iterations',type=int,default=3)
    parser.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--budget-seconds',type=float,default=0)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.iterations<1 or args.threads<1 or args.budget_seconds<0:
        parser.error('iterations, threads and budget must be positive')
    import cv2
    import numpy as np
    from .session import DemoSession
    cv2.setNumThreads(args.threads)
    fixture=Path(__file__).parent/'tests/fixtures/astronaut.png'
    session=DemoSession()
    session.load(fixture.read_bytes(),'.png')
    session.ensure_regions()
    session.ensure_aged_proposal(30,75,quality=args.quality)
    result=session.render_age(30,75)
    first_photo_seconds=perf_counter()-PROCESS_START
    stages={'source_detection_ms':session.detection_ms,'preparation_ms':session.preparation_ms,
        'semantic_parsing_ms':session.segmentation_ms,'conditional_skin_ms':session.proposal_ms,
        'compositing_ms':result.metrics['total_ms']}
    new_age_seconds=[]
    for i in range(args.iterations):
        age=76+i%3
        start=perf_counter()
        session.ensure_aged_proposal(30,age,quality=args.quality)
        session.render_age(30,age)
        new_age_seconds.append(perf_counter()-start)
    cached_start=perf_counter()
    cached=session.render_age(30,age)
    cache_ms=(perf_counter()-cached_start)*1000
    warm_p95=float(np.percentile(new_age_seconds,95))
    passed=(first_photo_seconds<=args.budget_seconds and max(new_age_seconds)<=args.budget_seconds) if args.budget_seconds else None
    record={'platform':platform.platform(),'python':platform.python_version(),
        'cpu':platform.processor(),'logical_cpus':os.cpu_count(),'opencv_threads':cv2.getNumThreads(),
        'quality':args.quality,'neural_forward_windows':session.proposal_metrics['forward_count'],
        'runtime_gpu_acceleration':False,'first_photo_seconds':round(first_photo_seconds,3),
        'first_photo_stage_ms':{k:round(v,2) for k,v in stages.items()},
        'new_age_seconds':[round(v,3) for v in new_age_seconds],
        'new_age_p95_seconds':round(warm_p95,3),'inspection_cache_ms':round(cache_ms,3),
        'source_detection_passes':session.detection_count,'semantic_parsing_passes':session.segmentation_count,
        'conditional_age_inferences':session.proposal_count,'working_shape':list(session.prepared.image.shape),
        'budget_seconds':args.budget_seconds,'budget_pass':passed,
        'scope':'Fresh Python process and first photo + new age changes after models installed. OS file cache may be warm. Excludes initial dependency/model downloads and browser display. Run on your actual laptop; cloud results do not prove laptop performance or visual age accuracy.'}
    payload=json.dumps(record,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload+'\n')
    print(payload)
    raise SystemExit(2 if passed is False else 0)

if __name__=='__main__':main()
