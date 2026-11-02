"""Offline authored-layer tests. No participant photos or model downloads."""
from pathlib import Path
from io import BytesIO
import numpy as np
import pytest
from ..hair_layers import (HairModelUnavailable,conservative_hair_mask,gray_hair,
    head_crop_bounds,model_candidates,segment_hair,segment_regions,verified_model_path)
from ..prepare_hair_model import ensure_model


@pytest.fixture
def synthetic_hair():
    h,w=96,112
    y,x=np.indices((h,w))
    luminance=40+((x//2)%2)*60+y*.25
    image=np.stack((luminance*1.3,luminance*.85,luminance*.55),axis=-1).clip(0,255).astype(np.uint8)
    mask=np.zeros((h,w),np.float32);mask[12:80,18:94]=1
    return image,mask


def test_zero_strength_exact_identity(synthetic_hair):
    image,mask=synthetic_hair
    result,metrics=gray_hair(image,mask,0)
    np.testing.assert_array_equal(result,image)
    assert not metrics['applied'] and metrics['mean_change']==0


def test_empty_or_uncertain_mask_does_not_invent_hair(synthetic_hair):
    image,_=synthetic_hair
    for value in (0,.2,.5):
        result,metrics=gray_hair(image,np.full(image.shape[:2],value,np.float32),1)
        np.testing.assert_array_equal(result,image)
        assert not metrics['applied']


def test_only_confident_hair_changes_and_input_unchanged(synthetic_hair):
    image,mask=synthetic_hair
    before=image.copy()
    result,metrics=gray_hair(image,mask,1)
    np.testing.assert_array_equal(result[mask==0],image[mask==0])
    np.testing.assert_array_equal(image,before)
    assert metrics['applied'] and metrics['hair_pixels']==int((mask>.5).sum())
    assert metrics['mean_change']>5


def test_deterministic_strands_reduce_chroma_and_preserve_contrast(synthetic_hair):
    image,mask=synthetic_hair
    a,_=gray_hair(image,mask,1,seed=3)
    b,_=gray_hair(image,mask,1,seed=3)
    c,_=gray_hair(image,mask,1,seed=4)
    np.testing.assert_array_equal(a,b)
    assert np.any(a[mask>0]!=c[mask>0])
    source=image[mask>0].astype(float)
    changed=a[mask>0].astype(float)
    assert np.ptp(changed,axis=1).mean()<np.ptp(source,axis=1).mean()*.5
    old_lum=source@np.array([.2126,.7152,.0722])
    new_lum=changed@np.array([.2126,.7152,.0722])
    assert np.std(new_lum)>np.std(old_lum)*.6
    assert np.corrcoef(old_lum,new_lum)[0,1]>.95
    assert new_lum.max()<245

@pytest.mark.parametrize('strength',[-.1,1.1,float('nan')])
def test_invalid_strength(synthetic_hair,strength):
    with pytest.raises(ValueError): gray_hair(*synthetic_hair,strength)


def test_invalid_masks_and_model(synthetic_hair,tmp_path):
    image,mask=synthetic_hair
    for bad in (mask[:2],mask+1,np.full_like(mask,np.nan)):
        with pytest.raises(ValueError): gray_hair(image,bad,.5)
    with pytest.raises(HairModelUnavailable,match='missing'):
        segment_hair(image,tmp_path/'absent.onnx')
    bad=tmp_path/'bad.onnx';bad.write_bytes(b'not-a-model')
    with pytest.raises(HairModelUnavailable,match='SHA-256'):
        verified_model_path(bad)


def test_installer_keeps_existing_unverified_file(tmp_path):
    path=tmp_path/'model.onnx';path.write_bytes(b'existing-user-file')
    with pytest.raises(ValueError,match='not overwritten'):
        ensure_model(path)
    assert path.read_bytes()==b'existing-user-file'


def test_download_hash_failure_removes_partial_and_never_installs(tmp_path,monkeypatch):
    import experiments.geometry_renderer.prepare_hair_model as installer
    monkeypatch.setattr(installer.urllib.request,'urlopen',lambda *args,**kwargs:BytesIO(b'wrong-model'))
    path=tmp_path/'model.onnx'
    with pytest.raises(ValueError,match='SHA-256'):
        ensure_model(path)
    assert not path.exists() and list(tmp_path.iterdir())==[]


def test_head_crop_is_fixed_bounded_and_excludes_outside(licensed_image,licensed_landmarks):
    bounds=head_crop_bounds(licensed_image,licensed_landmarks)
    x0,y0,x1,y1=bounds
    assert 0<=x0<x1<=licensed_image.shape[1] and 0<=y0<y1<=licensed_image.shape[0]
    assert y0<licensed_landmarks[:,1].min()
    assert head_crop_bounds(licensed_image,licensed_landmarks)==bounds
    fallback=conservative_hair_mask(licensed_image,licensed_landmarks)
    # Fallback explicitly protects central face even without a semantic parser.
    for index in (1,2,4,13,14,33,263):
        x,y=np.rint(licensed_landmarks[index]).astype(int)
        assert fallback[y,x]==0


def test_real_verified_offline_parser_on_licensed_fixture(licensed_image,licensed_landmarks,monkeypatch):
    candidates=[p for p in model_candidates() if p.is_file()]
    if not candidates:
        pytest.skip('Optional parser model is not installed; no runtime download attempted')
    import socket
    def forbidden(*args,**kwargs):
        raise AssertionError('Rendering must not use the network')
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    regions=segment_regions(licensed_image,candidates[0],landmarks=licensed_landmarks)
    assert regions['labels'].dtype==np.uint8
    for name in ('hair','skin','neck'):
        assert regions[name].shape==licensed_image.shape[:2]
        assert regions[name].dtype==np.float32
        assert np.isfinite(regions[name]).all()
        assert 0<=regions[name].min()<=regions[name].max()<=1
    assert (regions['hair']>.5).sum()>1000
    assert (regions['skin']>.5).sum()>1000
    assert not np.any((regions['skin']>.5)&(regions['hair']>.5))
    for index in (1,2,4,33,263):
        x,y=np.rint(licensed_landmarks[index]).astype(int)
        assert regions['hair'][y,x]==0
    changed,metrics=gray_hair(licensed_image,regions['hair'],1)
    assert metrics['applied'] and metrics['mean_change']>10
    np.testing.assert_array_equal(changed[regions['hair']<=.5],licensed_image[regions['hair']<=.5])
