"""Launch from repo root: python -m streamlit run experiments/geometry_renderer/app.py."""
from pathlib import Path
import sys
from time import perf_counter

# Streamlit executes a script by filename; support both documented root launch
# and direct script execution without installing this experiment as a package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from experiments.geometry_renderer.core import render, overlay
from experiments.geometry_renderer.session import DemoSession

st.set_page_config(page_title='AVATAR Geometry Renderer v0.1', layout='wide')
st.title('AVATAR Geometry Renderer v0.1')
st.warning('Illustrative geometric simulation — not a scientifically calibrated prediction.')
st.markdown('The nose, cheek and jowl controls are **authored hypotheses pending quantitative '
            'anthropometric calibration**. No texture aging, wrinkles, hair whitening, ears, '
            'GPT calls, or explanation of image-model internals.')
st.caption('Local CPU processing in session memory. Run this server on your own computer, '
           'bound to 127.0.0.1. No photos are sent to external services or automatically saved.')

if 'demo_session' not in st.session_state:
    st.session_state.demo_session = DemoSession()
if 'upload_generation' not in st.session_state:
    st.session_state.upload_generation = 0
session = st.session_state.demo_session


def reset_deformation():
    st.session_state.target_age = st.session_state.get('current_age', 30)
    for key in ('nose', 'cheeks', 'jowls'):
        st.session_state[key] = 0.0
    st.session_state.show_mesh = False


def clear_image():
    session.clear()
    st.session_state.use_sample = False
    st.session_state.upload_generation += 1


with st.sidebar:
    st.header('Local portrait')
    uploaded = st.file_uploader('Select JPEG or PNG (10 MiB maximum)', type=['jpg', 'jpeg', 'png'],
                               key=f'portrait_{st.session_state.upload_generation}')
    use_sample = st.checkbox('Use public-domain NASA sample', key='use_sample',
                             help='NASA / Eileen Collins via scikit-image. Not a participant portrait.')
    st.button('Clear image from this session', on_click=clear_image)
    st.header('Geometry hypotheses')
    current = st.slider('Current age', 18, 90, 30, key='current_age')
    if st.session_state.get('target_age', current) < current:
        st.session_state.target_age = current
    target = st.slider('Illustrative target age', current, 100, max(current, 65), key='target_age')
    nose = st.slider('Nasal tip ptosis — downward tip shift', 0.0, 1.0, .5, .05, key='nose')
    cheeks = st.slider('Cheek / midface descent', 0.0, 1.0, .5, .05, key='cheeks')
    jowls = st.slider('Lower-face / jowl sagging', 0.0, 1.0, .5, .05, key='jowls')
    mesh = st.checkbox('Show mesh and landmarks', key='show_mesh')
    st.button('Reset deformation', on_click=reset_deformation)

if uploaded is not None:
    data, suffix = uploaded.getvalue(), Path(uploaded.name).suffix
elif use_sample:
    data = (Path(__file__).parent / 'tests/fixtures/astronaut.png').read_bytes()
    suffix = '.png'
    st.caption('Public-domain NASA portrait of Eileen Collins, distributed by scikit-image. No NASA endorsement.')
else:
    session.clear()
    st.info('Select a local portrait with exactly one clearly visible face, or use the licensed sample.')
    st.stop()

try:
    prepared = session.load(data, suffix)
    start = perf_counter()
    result = render(prepared, current_age=current, target_age=target,
                    nose=nose, cheeks=cheeks, jowls=jowls)
    warp_ms = (perf_counter()-start)*1000
except ValueError as exc:
    # Validation errors are authored messages and do not include photo paths.
    st.error(str(exc))
    st.stop()
except Exception:
    session.clear()
    st.error('Local renderer could not initialize. Check the isolated dependencies; try a clearer portrait.')
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader(f'Original — stated current age {current}')
    st.image(overlay(prepared.image, prepared.points, prepared.triangles) if mesh else prepared.image,
             use_container_width=True)
with right:
    st.subheader(f'Illustrative geometry — selected age {target}')
    st.image(overlay(result.image, result.points, prepared.triangles) if mesh else result.image,
             use_container_width=True)
    st.caption('This is a geometric warp of the same image, not a generated or calibrated future portrait.')

if result.safety_scale < 1:
    st.info(f'Triangle safety limiter applied: {result.safety_scale:.3f} × requested displacement.')
st.caption(f'CPU detection: {session.detection_ms:.1f} ms · mesh preparation: {session.preparation_ms:.1f} ms · '
           f'last warp: {warp_ms:.1f} ms (excludes display/transport) · '
           f'{len(prepared.triangles)} triangles · detection passes for this image: {session.detection_count}. '
           'Sliders reuse the same landmarks and triangulation.')
st.caption('Clearing releases this session’s references; this is not secure memory erasure. '
           'Do not use the demo for participant collection or run it as a public server.')
