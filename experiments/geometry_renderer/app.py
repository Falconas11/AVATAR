"""Local age-baseline, learned skin appearance, geometry and segmented hair UI."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import os
import cv2
import streamlit as st
cv2.setNumThreads(min(4,os.cpu_count() or 1))
from experiments.geometry_renderer.core import overlay
from experiments.geometry_renderer.comparison import crop_view,difference_view,head_crop_bounds,wipe_view
from experiments.geometry_renderer.session import DemoSession

st.set_page_config(page_title='AVATAR Local Aging v0.2',layout='wide')
st.title('AVATAR Local Aging v0.2')
st.caption('Age baseline → face geometry + learned skin appearance + segmented gray hair. All photo processing stays on this computer.')
st.warning('Illustrative aging visualization. Target age is a control, not a calibrated prediction of how this person will look.')

if 'demo_session' not in st.session_state:
    st.session_state.demo_session = DemoSession()
if 'upload_generation' not in st.session_state:
    st.session_state.upload_generation = 0
session = st.session_state.demo_session


def reset_aging():
    st.session_state.target_age = st.session_state.get('current_age',30)
    st.session_state.show_mesh = False


def clear_image():
    session.clear()
    st.session_state.use_sample = False
    st.session_state.upload_generation += 1


with st.sidebar:
    st.header('Local portrait')
    uploaded = st.file_uploader('Select JPEG or PNG (10 MiB maximum)',type=['jpg','jpeg','png'],
        key=f'portrait_{st.session_state.upload_generation}')
    use_sample = st.checkbox('Use public-domain NASA sample',key='use_sample')
    st.button('Clear image from this session',on_click=clear_image)
    st.header('Age progression')
    current = st.slider('Current age',18,90,30,key='current_age')
    if st.session_state.get('target_age',current)<current:
        st.session_state.target_age=current
    target = st.slider('Illustrative target age',current,100,max(current,75),key='target_age')
    st.header('Aging layers')
    geometry_on = st.checkbox('Face geometry',value=True,key='geometry_on')
    skin_on = st.checkbox('Wrinkles and skin appearance',value=True,key='skin_on')
    quality_label = st.selectbox('Skin quality',['Detailed (4 windows)','Fast (1 window)'],key='skin_quality')
    quality = 'detail' if quality_label.startswith('Detailed') else 'fast'
    hair_on = st.checkbox('Hair graying',value=True,key='hair_on')
    skin_strength = st.slider('Skin appearance strength',0.,1.,1.,.05,key='skin_strength')
    hair_strength = st.slider('Hair graying strength',0.,1.,1.,.05,key='hair_strength')
    with st.expander('Geometry and lifestyle controls'):
        nose = st.slider('Nasal tip ptosis — downward tip shift',0.,1.,.65,.05,key='nose')
        cheeks = st.slider('Cheek / midface descent',0.,1.,.65,.05,key='cheeks')
        jowls = st.slider('Lower-face / jowl sagging',0.,1.,.65,.05,key='jowls')
        st.caption('Illustrative modifiers, not measured exposure or clinical risk. Zero modifiers still age.')
        uv = st.slider('UV modifier',0.,1.,0.,.1,key='uv_modifier')
        smoking = st.slider('Smoking modifier',0.,1.,0.,.1,key='smoking_modifier')
    st.header('Inspect changes')
    focus = st.checkbox('Focus on head and face',value=True,key='focus_face')
    view = st.selectbox('Comparison view',['Side by side','Before / after wipe','Pixel difference ×4'],key='comparison_view')
    mesh = st.checkbox('Show mesh and landmarks',key='show_mesh')
    inspect_masks = st.checkbox('Inspect hair and skin masks',key='inspect_masks')
    st.button('Reset aging',on_click=reset_aging)

if uploaded is not None:
    data,suffix = uploaded.getvalue(),Path(uploaded.name).suffix
elif use_sample:
    data = (Path(__file__).parent/'tests/fixtures/astronaut.png').read_bytes()
    suffix = '.png'
    st.caption('Sample: public-domain NASA portrait of Eileen Collins via scikit-image. Stated input age is illustrative; no endorsement is implied.')
else:
    session.clear()
    st.info('Select a local portrait with exactly one clearly visible face, or use the licensed sample.')
    st.stop()

try:
    prepared = session.load(data,suffix)
    regions = session.ensure_regions() if (skin_on or hair_on or inspect_masks) else (session.regions or {})
    if skin_on and target>current and skin_strength>0:
        session.ensure_aged_proposal(current,target,quality=quality)
    enabled = tuple(name for name,on in [('geometry',geometry_on),('skin',skin_on),('hair',hair_on)] if on)
    result = session.render_age(current,target,lifestyle={'uv_exposure':uv,'smoking':smoking},
        geometry_strengths={'nose':nose,'cheeks':cheeks,'jowls':jowls},
        skin_strength=skin_strength,hair_strength=hair_strength,enabled_layers=enabled)
except ValueError as exc:
    st.error(str(exc))
    st.stop()
except Exception:
    session.clear()
    st.error('The local aging renderer could not initialize. Check the isolated model setup and choose a clear frontal portrait.')
    st.stop()

original = overlay(prepared.image,prepared.points,prepared.triangles) if mesh else prepared.image
aged = overlay(result.image,result.points,prepared.triangles) if mesh else result.image
bounds = head_crop_bounds(prepared) if focus else (0,0,prepared.image.shape[1],prepared.image.shape[0])
original_view,aged_view = crop_view(original,bounds),crop_view(aged,bounds)
if view=='Side by side':
    left,right=st.columns(2)
    with left:
        st.subheader(f'Original — stated age {current}')
        st.image(original_view,use_container_width=True)
    with right:
        st.subheader(f'Aging visualization — target {target}')
        st.image(aged_view,use_container_width=True)
elif view=='Before / after wipe':
    st.subheader('Aligned before / after comparison')
    split=st.slider('Wipe position — original left, aged right',0,100,50,key='wipe_position')
    st.image(wipe_view(original_view,aged_view,split),use_container_width=True)
else:
    st.subheader('Pixel difference ×4 — diagnostic view')
    st.image(difference_view(crop_view(prepared.image,bounds),crop_view(result.image,bounds)),use_container_width=True)
    st.caption('Amplified RGB difference is an inspection tool, not a portrait or proof of perceived age.')

processing_ms = session.detection_ms+session.preparation_ms+session.segmentation_ms+session.proposal_ms+result.metrics['total_ms']
if processing_ms>20_000 and skin_on:
    st.warning('This computer exceeded the 20-second processing budget. Select Fast skin quality or turn off the optional skin appearance layer; mesh and hair rendering remain available.')
if current==target:
    st.info('Current and target age are equal: no additional aging is applied.')
if result.safety_scale<1:
    st.info(f'Geometry safety limiter: {result.safety_scale:.3f} × requested displacement.')
if hair_on and result.metrics['hair_mask_pixels']==0:
    st.info('No confident hair region was detected. Hair graying is unavailable for this image.')

with st.expander('What changed? Compare the layers'):
    columns=st.columns(3)
    for column,image,label in zip(columns,[result.geometry_image,result.skin_image,result.image],
        ['Geometry only','Geometry + skin','All enabled layers']):
        column.caption(label)
        column.image(crop_view(image,bounds),use_container_width=True)
    st.write({'geometry_changed_pixels':result.metrics['geometry_changed_pixels'],
        'skin_changed_pixels':result.metrics['skin_changed_pixels'],
        'hair_changed_pixels':result.metrics['hair_changed_pixels']})
    st.caption('Counts confirm that operators ran. They do not certify that a face looks a particular age.')

if inspect_masks:
    st.subheader('Source masks — inspect before interpreting the result')
    mask_columns=st.columns(2)
    for column,name in zip(mask_columns,['skin','hair']):
        column.caption(name.title())
        column.image(crop_view((regions[name]*255).astype('uint8'),bounds),use_container_width=True)
    for warning in regions.get('warnings',[]):
        st.caption(warning)

with st.expander('Knowledge graph, age baseline and evidence limits'):
    st.caption('Existing v03 graph paths supply reference context. Smooth age baselines and missing-feature overlays are provisional visual assumptions; they do not explain neural model internals.')
    rows=[]
    for name,record in result.plan['features'].items():
        rows.append({'Feature':name.replace('_',' '),'Age-baseline intensity':round(record['score'],3),
            'Graph status':record['graph_support_status'],'Baseline status':record['evidence_status']})
    st.dataframe(rows,use_container_width=True,hide_index=True)
    feature_id=st.selectbox('Inspect feature references',list(result.plan['features']),key='feature_reference')
    feature=result.plan['features'][feature_id]
    for assumption in feature['assumptions']:
        st.caption(assumption)
    for path in feature['graph_paths']:
        st.write(' → '.join(path['nodes']))
        st.caption(path['evidence_status'])
    for citation in feature['citations']:
        st.caption('Repository reference label (not independently verified): '+citation)
    st.json(result.plan['metadata'])

st.caption(f'Local render: {result.metrics["total_ms"]:.0f} ms · source detection passes: {session.detection_count} · '
    f'semantic parsing passes: {session.segmentation_count} · learned appearance passes: {session.proposal_count}. '
    'Age changes rerun the local skin model; detection/masks stay cached. Inspection controls reuse the last rendered image.')
st.caption('No photos are uploaded to external services, automatically saved, or sent to the research/Box pipeline. '
    'Clear removes this session’s retained references, not guaranteed secure memory erasure.')
