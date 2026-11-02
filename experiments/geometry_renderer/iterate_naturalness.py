"""Explicit synthetic-fixture experiments with reused neural proposals.

This diagnostic writes only the manifest-pinned fictional fixture outputs. Never
accepts arbitrary participant paths. Neural proposals cache outside the checkout
for matched post-render experiments, keyed by input/model hashes and target age.
"""
from pathlib import Path
import argparse,json,os
from time import perf_counter
from hashlib import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--reference',choices=('v03','v040'),default='v03')
    args=parser.parse_args()
    if not args.name.replace('_','').replace('-','').isalnum():parser.error('simple experiment name required')
    import cv2,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    from .session import DemoSession
    from .pipeline import render_local_aging
    from .comparison import head_crop_bounds,crop_view
    from .conditional_skin import MODEL_SHA256
    from .validate_synthetic_batch import load_manifest
    cv2.setNumThreads(4)
    root=Path(__file__).parent;fixtures=root/'tests/fixtures/synthetic_portraits'
    manifest=load_manifest(fixtures)
    cache=Path(os.environ.get('AVATAR_SYNTHETIC_CACHE',str(Path.home()/'.cache/avatar/synthetic-iterations')));cache.mkdir(parents=True,exist_ok=True)
    output=root/'validation/naturalness_iterations'/args.name;output.mkdir(parents=True,exist_ok=True)
    sheet=Image.new('RGB',(1536,64+560*4),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=22)
    reference_label='v0.4.0 released' if args.reference=='v040' else 'v0.3 reference'
    for x,title in enumerate(('Synthetic input',reference_label,'Current candidate')):draw.text((x*512+10,18),title,fill='black',font=font)
    report={'experiment':args.name,'reference':args.reference,'target':75,'source_age':30,'samples':[],
        'source_code_sha256':{n:sha256((root/n).read_bytes()).hexdigest() for n in ('hair_layers.py','mechanism_layers.py','skin_layers.py','pipeline.py')}}
    for index,item in enumerate(manifest['samples']):
        start=perf_counter();session=DemoSession();prepared=session.load((fixtures/item['file']).read_bytes(),'.png');regions=session.ensure_regions()
        key=sha256((item['sha256']+MODEL_SHA256+'30-75-ultra').encode()+prepared.image.tobytes()+prepared.points.tobytes()).hexdigest()
        cached=cache/f'{key}.png'
        cache_hit=cached.is_file()
        if cache_hit:proposal=np.asarray(Image.open(cached).convert('RGB')).copy()
        else:
            proposal=session.ensure_aged_proposal(30,75,quality='ultra');Image.fromarray(proposal).save(cached)
        result=render_local_aging(prepared,regions,30,75,lifestyle={'uv_exposure':0.,'smoking':0.},learned_proposal=proposal)
        bounds=head_crop_bounds(prepared);before=crop_view(prepared.image,bounds);after=crop_view(result.image,bounds)
        Image.fromarray(after).save(output/f'{item["id"]}_age75.png')
        reference_dir='validation/naturalness_iterations/source_locks_D' if args.reference=='v040' else 'validation/synthetic_v03'
        reference=np.asarray(Image.open(root/reference_dir/f'{item["id"]}_age75.png').convert('RGB'))
        row=64+index*560
        for column,pixels in enumerate((before,reference,after)):
            pic=Image.fromarray(pixels);pic.thumbnail((500,500),Image.Resampling.LANCZOS);sheet.paste(pic,(column*512+(512-pic.width)//2,row+32))
        draw.text((10,row+5),item['id'],fill='black',font=font)
        record={'id':item['id'],'safety_scale':result.safety_scale,'seconds':round(perf_counter()-start,3),'proposal_cache_hit':cache_hit,'hair':result.metrics['hair']}
        report['samples'].append(record);print(json.dumps(record),flush=True)
    sheet.save(output/'contact-sheet.jpg',quality=96)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
