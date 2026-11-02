"""Matched scenario validation on manifest-pinned fictional fixtures only.

Explicit CLI writes these synthetic outputs; interactive rendering saves none.
A single shared age proposal is reused for all scenarios of an input.
"""
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import argparse,json,os


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'validation/lifestyle_v01')
    args=parser.parse_args()
    import cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from .session import DemoSession
    from .conditional_skin import MODEL_SHA256
    from .lifestyle import PRESETS
    from .comparison import crop_view,head_crop_bounds
    from .validate_synthetic_batch import load_manifest
    cv2.setNumThreads(4)
    root=Path(__file__).parent;fixtures=root/'tests/fixtures/synthetic_portraits'
    manifest=load_manifest(fixtures);args.output.mkdir(parents=True,exist_ok=True)
    scenarios={k:v for k,v in PRESETS.items() if k!='Baseline (no added modifiers)'}
    labels=['Baseline']+list(scenarios)
    sheet=Image.new('RGB',(360*len(labels),64+420*4),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=18)
    for i,label in enumerate(labels):draw.text((i*360+8,20),label,fill='black',font=font)
    report={'model_sha256':MODEL_SHA256,'current_age':30,'target_age':75,'quality':'ultra','scenarios':scenarios,
        'controls':'same default geometry, skin/hair, mechanism/feature controls; no per-input tuning',
        'input_provenance':manifest['provenance'],'samples':[],
        'source_sha256':{n:sha256((root/n).read_bytes()).hexdigest() for n in ('lifestyle.py','age_plan.py','pipeline.py','skin_layers.py','session.py','app.py')},
        'limitations':'Qualitative scenarios with unverified graph references and authored gains, not clinical exposure/duration or perceptual validation. Cached proposal times exclude inference. No laptop performance claim.'}
    cache=Path(os.environ.get('AVATAR_SYNTHETIC_CACHE','/tmp/avatar-synthetic-cache'));cache.mkdir(parents=True,exist_ok=True)
    for index,item in enumerate(manifest['samples']):
        s=DemoSession();prepared=s.load((fixtures/item['file']).read_bytes(),'.png');s.ensure_regions()
        key=sha256((item['sha256']+MODEL_SHA256+'30-75-ultra').encode()+prepared.image.tobytes()+prepared.points.tobytes()).hexdigest()
        path=cache/f'{key}.png';hit=path.is_file();start=perf_counter()
        if hit:
            proposal=np.asarray(Image.open(path).convert('RGB')).copy()
            s.ensure_aged_proposal(30,75,quality='ultra',generator=lambda *a,**k:(proposal,{'fixture_proposal_cache_hit':True}))
        else:
            proposal=s.ensure_aged_proposal(30,75,quality='ultra');Image.fromarray(proposal).save(path)
        record={'id':item['id'],'input_sha256':item['sha256'],'proposal_cache_hit':hit,
            'proposal_seconds':round(perf_counter()-start,3),'scenarios':{}}
        base=s.render_age(30,75);bounds=head_crop_bounds(prepared)
        baseline=crop_view(base.image,bounds);panels=[baseline]
        Image.fromarray(baseline).save(args.output/f'{item["id"]}_baseline.png')
        for label,values in scenarios.items():
            start=perf_counter();result=s.render_age(30,75,lifestyle=values)
            difference=np.abs(result.image.astype(float)-base.image.astype(float))
            panel=crop_view(result.image,bounds);panels.append(panel)
            slug=label.lower().split(' (')[0].replace(' ','_')
            Image.fromarray(panel).save(args.output/f'{item["id"]}_{slug}.png')
            record['scenarios'][label]={'seconds':round(perf_counter()-start,3),
                'changed_pixels':int(np.any(result.image!=base.image,axis=2).sum()),
                'mean_abs_rgb_difference':round(float(difference.mean()),4),
                'safety_scale':result.safety_scale,'hair_score_unchanged':result.plan['features']['hair_graying']['score']==base.plan['features']['hair_graying']['score'],
                'requested_effects':{k:list(v['features']) for k,v in result.plan['lifestyle_scenarios'].items() if v['features']}}
            if not np.any(difference):raise AssertionError(f'No visible pixel difference: {item["id"]} {label}')
        diet=s.render_age(30,75,lifestyle={'diet':1})
        np.testing.assert_array_equal(diet.image,base.image)
        record.update(diet_exact_baseline=True,detection_passes=s.detection_count,parsing_passes=s.segmentation_count,proposal_preparations=s.proposal_count,
            neural_inference_runs=0 if hit else 1)
        if (s.detection_count,s.segmentation_count,s.proposal_count)!=(1,1,1):raise AssertionError('Scenario reran shared preparation')
        # Verify the no-exposure output is unchanged from the previous revision.
        reference=np.asarray(Image.open(root/'validation/naturalness_iterations/source_locks_D'/f'{item["id"]}_age75.png').convert('RGB'))
        record['baseline_matches_previous_revision']=bool(np.array_equal(baseline,reference))
        previous_delta=np.abs(baseline.astype(float)-reference.astype(float))
        record['previous_baseline_difference']={'pixels':int(np.any(previous_delta,axis=2).sum()),'max_rgb':float(previous_delta.max()),'mean_rgb':float(previous_delta.mean())}
        for col,pixels in enumerate(panels):
            pic=Image.fromarray(pixels);pic.thumbnail((350,380),Image.Resampling.LANCZOS)
            sheet.paste(pic,(col*360+(360-pic.width)//2,64+index*420+28))
        draw.text((10,64+index*420+5),item['id'],fill='black',font=font)
        report['samples'].append(record);print(json.dumps(record),flush=True)
    sheet.save(args.output/'contact-sheet.jpg',quality=96)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
