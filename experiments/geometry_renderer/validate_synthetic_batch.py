"""Explicit fixture-only batch validation. Never calls an image-generation API.

Inputs are four synthetic portraits with pinned hashes. Only an explicit CLI run
writes their outputs/contact sheet. The interactive application still saves no photos.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter


def load_manifest(folder):
    data=json.loads((folder/'manifest.json').read_text())
    for sample in data['samples']:
        path=folder/sample['file']
        if path.parent!=folder or sha256(path.read_bytes()).hexdigest()!=sample['sha256']:
            raise ValueError('Synthetic fixture path/digest mismatch')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quality',choices=('fast','detail','ultra'),default='ultra')
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'validation/synthetic_v03')
    args=parser.parse_args()
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from .session import DemoSession
    from .comparison import head_crop_bounds,crop_view
    cv2.setNumThreads(4)
    folder=Path(__file__).parent/'tests/fixtures/synthetic_portraits'
    manifest=load_manifest(folder);args.output.mkdir(parents=True,exist_ok=True)
    controls={'current_age':30,'target_ages':[60,75],'quality':args.quality,
        'geometry_strengths':{'nose':.65,'cheeks':.65,'jowls':.65},'skin_strength':1.,'hair_strength':1.,
        'lifestyle':{'smoking':0.,'uv_exposure':0.},'all_mechanism_controls':1.,'per_image_tuning':False}
    report={'controls':controls,'provenance':manifest['provenance'],'age_note':manifest['age_note'],
        'platform':platform.platform(),'opencv_threads':4,'samples':[]}
    report['renderer_code_sha256']={name:sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ('pipeline.py','age_plan.py','conditional_skin.py','skin_layers.py','hair_layers.py','core.py','session.py','detection.py')}
    try:report['commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:report['commit']=None
    font=ImageFont.load_default(size=22);small=ImageFont.load_default(size=17)
    sheet=Image.new('RGB',(1536,64+len(manifest['samples'])*560),'white');draw=ImageDraw.Draw(sheet)
    for col,title in enumerate(('Synthetic input / control30','Program output / target60','Program output / target75')):
        draw.text((col*512+12,18),title,fill='black',font=font)
    for index,sample in enumerate(manifest['samples']):
        start=perf_counter();record={'id':sample['id'],'input_sha256':sample['sha256']}
        try:
            session=DemoSession();prepared=session.load((folder/sample['file']).read_bytes(),'.png')
            session.ensure_regions();bounds=head_crop_bounds(prepared)
            original=crop_view(prepared.image,bounds);panels=[original]
            face=prepared.points[:468];native_height=float(np.ptp(face[:,1])/session.working_scale)
            record.update(input_size=sample['size'],native_face_height_px=round(native_height,1),
                working_face_height_px=round(float(np.ptp(face[:,1])),1),working_scale=session.working_scale,
                source_crop_bounds=list(bounds),targets={})
            for age in controls['target_ages']:
                tick=perf_counter();session.ensure_aged_proposal(30,age,quality=args.quality)
                result=session.render_age(30,age,lifestyle=controls['lifestyle'],geometry_strengths=controls['geometry_strengths'])
                view=crop_view(result.image,bounds);panels.append(view)
                path=args.output/f'{sample["id"]}_age{age}.png';Image.fromarray(view).save(path)
                record['targets'][str(age)]={'seconds':round(perf_counter()-tick,3),'output_sha256':sha256(path.read_bytes()).hexdigest(),
                    'safety_scale':result.safety_scale,'ear_availability':result.metrics['ear_warp']['available'],
                    'source_feature_count':len(result.metrics['feature_coverage']),'mechanism_count':len(result.metrics['mechanisms']),
                    'skin_changed_pixels':result.metrics['skin_changed_pixels'],'hair_changed_pixels':result.metrics['hair_changed_pixels'],
                    'hand_authored_crease_paths_used':result.metrics['skin'].get('hand_authored_crease_paths_used'),
                    'skin_backend':session.proposal_metrics['backend']}
            Image.fromarray(original).save(args.output/f'{sample["id"]}_input.png')
            row=64+index*560
            for col,panel in enumerate(panels):
                image=Image.fromarray(panel);image.thumbnail((500,500),Image.Resampling.LANCZOS)
                sheet.paste(image,(col*512+(512-image.width)//2,row+30))
            draw.text((12,row+5),f'{sample["id"]}  native face {native_height:.0f}px',fill='black',font=small)
            pair=Image.new('RGB',(1024,550),'white');pd=ImageDraw.Draw(pair)
            for col,(panel,title) in enumerate(((panels[0],'Synthetic input'),(panels[2],'Local renderer / target75'))):
                pic=Image.fromarray(panel);pic.thumbnail((500,500),Image.Resampling.LANCZOS)
                pair.paste(pic,(col*512+(512-pic.width)//2,40));pd.text((col*512+10,10),title,fill='black',font=font)
            pair.save(args.output/f'{sample["id"]}_comparison.jpg',quality=96)
            record.update(status='completed',elapsed_seconds=round(perf_counter()-start,3),
                detection_passes=session.detection_count,parsing_passes=session.segmentation_count,appearance_passes=session.proposal_count)
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        report['samples'].append(record)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'sample':sample['id'],'status':record['status'],'elapsed':record.get('elapsed_seconds'),'error':record.get('error')},ensure_ascii=False),flush=True)
    sheet.save(args.output/'contact-sheet.jpg',quality=96)
    report['completed']=sum(s['status']=='completed' for s in report['samples'])
    report['failed']=len(report['samples'])-report['completed']
    report['limitations']='Four synthetic inputs under matched conditions; not a real-world, demographic, identity-preservation or exact-age accuracy study. All aged outputs are produced by repository code, not the image-generation tool.'
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed':report['completed'],'failed':report['failed']}),flush=True)
    raise SystemExit(1 if report['failed'] else 0)

if __name__=='__main__':main()
