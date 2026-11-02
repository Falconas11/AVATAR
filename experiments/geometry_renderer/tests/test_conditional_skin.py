"""Optional verified local skin model; fixture-only offline checks."""
from hashlib import sha256
import json
from pathlib import Path
import socket
import os
import numpy as np
import pytest
from experiments.geometry_renderer.conditional_skin import (MODEL_BYTES,MODEL_SHA256,
    ConditionalModelUnavailable,generate_aged_skin,verified_model_path,_windowed_residual)


_CANDIDATES=[Path.home()/'.cache/avatar/fran_reaging.onnx',Path('/workspace/avatar-runtime/models/fran_reaging.onnx')]
if os.environ.get('AVATAR_CONDITIONAL_AGING_MODEL'):
    _CANDIDATES.insert(0,Path(os.environ['AVATAR_CONDITIONAL_AGING_MODEL']).expanduser())
_HAS_OPTIONAL_MODEL=any(path.is_file() for path in _CANDIDATES)


def test_optional_model_absent_or_tampered_fails_closed(tmp_path):
    with pytest.raises(ConditionalModelUnavailable,match='missing'):
        verified_model_path(tmp_path/'missing.onnx')
    path=tmp_path/'changed.onnx';path.write_bytes(b'changed')
    with pytest.raises(ConditionalModelUnavailable,match='SHA-256'):
        verified_model_path(path)


def test_equal_age_identity_without_model(licensed_image,licensed_landmarks,tmp_path):
    output,metrics=generate_aged_skin(licensed_image,licensed_landmarks,current_age=65,target_age=65,
                                    model_path=tmp_path/'not-installed.onnx')
    np.testing.assert_array_equal(output,licensed_image)
    assert not metrics['applied'] and metrics['inference_ms']==0


def test_optional_manifest_matches_verified_artifact():
    manifest=json.loads((Path(__file__).resolve().parents[1]/'models/conditional_skin.json').read_text())
    assert manifest['artifact_sha256']==MODEL_SHA256 and manifest['artifact_bytes']==MODEL_BYTES
    if _HAS_OPTIONAL_MODEL:
        path=verified_model_path();assert sha256(path.read_bytes()).hexdigest()==MODEL_SHA256


@pytest.mark.skipif(not _HAS_OPTIONAL_MODEL,reason='Optional FRAN model not installed; run explicit setup to exercise real inference')
def test_real_conditional_proposal_offline_and_crop_bounded(licensed_image,licensed_landmarks,monkeypatch):
    before=licensed_image.copy()
    def no_network(*args,**kwargs):raise AssertionError('Photo processing cannot access the network')
    monkeypatch.setattr(socket.socket,'connect',no_network)
    output,metrics=generate_aged_skin(licensed_image,licensed_landmarks,current_age=30,target_age=75)
    assert output.shape==before.shape and output.dtype==np.uint8
    assert metrics['target_age_condition']==75 and metrics['clinically_calibrated'] is False
    assert metrics['applied'] and metrics['mean_head_change']>2
    x0,y0,x1,y1=metrics['head_crop'];outside=np.ones(before.shape[:2],bool);outside[y0:y1,x0:x1]=False
    np.testing.assert_array_equal(output[outside],before[outside]);np.testing.assert_array_equal(licensed_image,before)


@pytest.mark.parametrize('source,target',[(17,75),(30,101),(65,30),(30.5,75),(True,75)])
def test_invalid_age_conditions(licensed_image,licensed_landmarks,source,target):
    with pytest.raises(ValueError,match='integer ages'):
        generate_aged_skin(licensed_image,licensed_landmarks,current_age=source,target_age=target)


@pytest.mark.parametrize('size,expected',[(512,1),(768,4)])
def test_overlap_windows_preserve_constant_residual_and_age_channels(size,expected):
    class ConstantNetwork:
        calls=0
        def setInput(self,tensor):
            assert tensor.shape==(1,5,512,512)
            np.testing.assert_allclose(tensor[:,3],.30)
            np.testing.assert_allclose(tensor[:,4],.75)
            self.calls+=1
        def forward(self):
            return np.full((1,3,512,512),.125,np.float32)
    network=ConstantNetwork()
    delta,count=_windowed_residual(network,np.zeros((size,size,3),np.float32),30,75)
    assert count==expected and network.calls==expected
    np.testing.assert_allclose(delta,.125,atol=1e-6)


@pytest.mark.skipif(not _HAS_OPTIONAL_MODEL,reason='Optional FRAN model not installed; run explicit setup for real detail inference')
def test_real_detail_uses_four_windows(licensed_image,licensed_landmarks):
    output,metrics=generate_aged_skin(licensed_image,licensed_landmarks,current_age=30,target_age=75,quality='detail')
    assert metrics['forward_count']==4 and metrics['working_size']==768
    assert metrics['mean_head_change']>2 and np.any(output!=licensed_image)


def test_unsupported_detail_mode_rejected(licensed_image,licensed_landmarks):
    with pytest.raises(ValueError,match='quality'):
        generate_aged_skin(licensed_image,licensed_landmarks,current_age=30,target_age=75,quality='slow-nine-window')
