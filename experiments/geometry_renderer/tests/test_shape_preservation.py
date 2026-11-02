"""Source contour regressions on generated faces; no dataset/user photographs."""
from pathlib import Path
import numpy as np
import pytest
from experiments.geometry_renderer.session import DemoSession
from experiments.geometry_renderer.shape_preservation import face_frame,shape_ratios,preserve_source_shape,LIMITS
from experiments.geometry_renderer.pipeline import render_local_aging
from experiments.geometry_renderer.mechanism_layers import volume_displacement
from experiments.geometry_renderer.age_plan import build_age_plan
from experiments.geometry_renderer.core import signed_areas


@pytest.fixture(scope='module')
def fictional_faces():
    root=Path(__file__).parent/'fixtures/synthetic_portraits'
    return [DemoSession().load((root/f'S0{i}.png').read_bytes(),'.png') for i in range(1,5)]


@pytest.mark.parametrize('index',range(4))
def test_realistic_contours_keep_source_ratios_and_positive_triangles(fictional_faces,index):
    p=fictional_faces[index]
    for age in (60,75,95):
        result=render_local_aging(p,None,30,age,enabled_layers=('geometry',))
        record=result.metrics['shape_preservation'];src=record['source'];after=record['rendered']
        assert np.any(result.points!=p.points)
        for key,bound in LIMITS.items():assert abs(after[key]/src[key]-1)<=bound+1e-7
        assert (signed_areas(result.points,p.triangles)>=signed_areas(p.points,p.triangles)*.1-1e-8).all()
        np.testing.assert_array_equal(result.points[468:],p.points[468:])


def test_deliberate_jaw_widening_is_limited_without_erasing_source_shapes(fictional_faces):
    ratios=[]
    for p in fictional_faces:
        lateral,_=face_frame(p.points);delta=np.zeros_like(p.points);fw=shape_ratios(p.points)['cheek_width']
        for left,right in ((136,365),(172,397)):
            delta[left]-=lateral*fw*.12;delta[right]+=lateral*fw*.12
        corrected,record=preserve_source_shape(p.points,delta)
        assert 0<record['scale']<1
        source=record['source']['jaw_to_cheek'];after=record['guarded']['jaw_to_cheek']
        assert abs(after/source-1)<=.015+1e-8
        ratios.append(source)
    assert np.ptp(ratios)>.01  # source faces were not replaced by one mean shape


def test_rotation_translation_equivariance_and_zero_field(fictional_faces):
    p=fictional_faces[0].points;plan=build_age_plan(30,75)
    delta=volume_displacement(p,468,plan)
    angle=.35;rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    moved=p@rot.T+[100,30]
    other=volume_displacement(moved,468,plan)
    np.testing.assert_allclose(other,delta@rot.T,atol=1e-8)
    a,one=preserve_source_shape(p,delta);b,two=preserve_source_shape(moved,other)
    np.testing.assert_allclose(b,a@rot.T,atol=1e-7)
    zero,record=preserve_source_shape(p,np.zeros_like(p))
    assert not zero.any() and record['scale']==1


def test_nose_slider_does_not_activate_cheek_and_jowl_volume(fictional_faces):
    p=fictional_faces[0]
    result=render_local_aging(p,None,30,75,enabled_layers=('geometry',),geometry_strengths={'nose':1.,'cheeks':0.,'jowls':0.})
    # The nose's compact radial field does not reach the jaw corners.
    np.testing.assert_allclose(result.points[[172,397,136,365]],p.points[[172,397,136,365]],atol=1e-8)
