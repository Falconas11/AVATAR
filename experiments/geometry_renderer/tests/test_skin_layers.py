"""Offline texture tests with a licensed fixture and deterministic synthetic RGB."""
import numpy as np
import cv2
import pytest
from experiments.geometry_renderer.skin_layers import (SUPPORTED_FEATURES,FACE_OVAL,
    LEFT_EYE,RIGHT_EYE,LIPS,LEFT_BROW,RIGHT_BROW,render_skin,skin_region_mask)


def plan(score):
    return {'features':{name:{'score':score} for name in SUPPORTED_FEATURES}}


def test_zero_scores_are_pixel_exact(licensed_image,licensed_landmarks):
    out,metrics = render_skin(licensed_image,licensed_landmarks,plan(0))
    np.testing.assert_array_equal(out,licensed_image)
    assert out is not licensed_image
    assert metrics['changed_pixels'] == 0


def test_visible_deterministic_effect_and_no_input_mutation(licensed_image,licensed_landmarks):
    before = licensed_image.copy()
    first,metrics = render_skin(licensed_image,licensed_landmarks,plan(.85))
    second,_ = render_skin(licensed_image,licensed_landmarks,plan(.85))
    np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(licensed_image,before)
    assert first.dtype == np.uint8 and first.shape == licensed_image.shape
    assert metrics['changed_pixels'] > 1000
    mask = skin_region_mask(licensed_image.shape,licensed_landmarks)
    assert np.abs(first.astype(float)-licensed_image)[mask>.5].mean() > 1
    assert set(metrics['features_rendered']) == set(SUPPORTED_FEATURES)


def test_eyes_lips_brows_and_background_remain_exact(licensed_image,licensed_landmarks):
    out,_ = render_skin(licensed_image,licensed_landmarks,plan(1))
    mask = skin_region_mask(licensed_image.shape,licensed_landmarks)
    np.testing.assert_array_equal(out[mask==0],licensed_image[mask==0])
    protected = np.zeros(licensed_image.shape[:2],np.uint8)
    for ids in (LEFT_EYE,RIGHT_EYE,LIPS,LEFT_BROW,RIGHT_BROW):
        cv2.fillConvexPoly(protected,cv2.convexHull(np.rint(licensed_landmarks[list(ids)]).astype(np.int32)),1)
    assert protected.any()
    np.testing.assert_array_equal(out[protected>0],licensed_image[protected>0])


def test_external_mask_can_disable_all_or_part(licensed_image,licensed_landmarks):
    zeros = np.zeros(licensed_image.shape[:2],np.float32)
    out,metrics = render_skin(licensed_image,licensed_landmarks,plan(1),skin_mask=zeros)
    np.testing.assert_array_equal(out,licensed_image)
    assert metrics['changed_pixels'] == 0
    allowed = np.zeros_like(zeros)
    allowed[:,:licensed_image.shape[1]//2] = 1
    out,_ = render_skin(licensed_image,licensed_landmarks,plan(1),skin_mask=allowed)
    np.testing.assert_array_equal(out[allowed==0],licensed_image[allowed==0])
    assert np.any(out[allowed>0] != licensed_image[allowed>0])


@pytest.mark.parametrize('feature',SUPPORTED_FEATURES)
def test_individual_layers_change_a_usable_face(licensed_image,licensed_landmarks,feature):
    out,metrics = render_skin(licensed_image,licensed_landmarks,{'features':{feature:{'score':1}}})
    assert metrics['changed_pixels'] > 0
    assert np.any(out != licensed_image)


def test_invalid_mask_and_scores_rejected(licensed_image,licensed_landmarks):
    with pytest.raises(ValueError):
        render_skin(licensed_image,licensed_landmarks,{'features':{'skin_texture':{'score':float('nan')}}})
    with pytest.raises(ValueError):
        render_skin(licensed_image,licensed_landmarks,plan(1),skin_mask=np.ones((4,4)))


def test_neck_lines_require_semantic_neck_mask(licensed_image,licensed_landmarks):
    neck_plan = {'features':{'neck_lines':{'score':1}}}
    out,metrics = render_skin(licensed_image,licensed_landmarks,neck_plan)
    np.testing.assert_array_equal(out,licensed_image)
    assert metrics['unsupported_features'] == ['neck_lines']
    h,w = licensed_image.shape[:2]
    y,x = np.ogrid[:h,:w]
    p = licensed_landmarks
    width,height = np.ptp(p,axis=0)
    neck = ((y >= p[152,1]) & (y < p[152,1]+height*.5) &
            (abs(x-p[152,0]) < width*.3)).astype(np.float32)
    out,metrics = render_skin(licensed_image,licensed_landmarks,neck_plan,neck_mask=neck)
    assert np.any(out != licensed_image)
    np.testing.assert_array_equal(out[neck==0],licensed_image[neck==0])
    assert metrics['features_rendered'] == ['neck_lines']


def test_learned_proposal_uses_no_authored_creases(licensed_image,licensed_landmarks,monkeypatch):
    import experiments.geometry_renderer.skin_layers as module
    def forbidden(*args,**kwargs):
        raise AssertionError('Authored crease paths must not run for learned proposals')
    monkeypatch.setattr(module,'_wrinkle_paths',forbidden)
    yy,xx = np.indices(licensed_image.shape[:2])
    lab = cv2.cvtColor(licensed_image,cv2.COLOR_RGB2LAB)
    # Synthetic unit-test residual only; this does not stand in for visual aging
    # validation, which must inspect the actual trained model's output separately.
    modulated = lab.copy()
    modulation = 16*np.sin(xx*.38)+9*np.cos(yy*.17)
    modulated[...,0] = np.clip(lab[...,0].astype(float)+modulation,0,255).astype(np.uint8)
    proposal = cv2.cvtColor(modulated,cv2.COLOR_LAB2RGB)
    before = licensed_image.copy()
    output,metrics = render_skin(licensed_image,licensed_landmarks,plan(.9),learned_proposal=proposal)
    assert metrics['changed_pixels'] > 1000
    assert metrics['applied_features'] == ['learned_skin_appearance']
    assert metrics['requested_features'] == list(SUPPORTED_FEATURES)
    assert metrics['source_chroma_preserved']
    assert not metrics['hand_authored_crease_paths_used']
    protected = skin_region_mask(licensed_image.shape,licensed_landmarks)==0
    np.testing.assert_array_equal(output[protected],licensed_image[protected])
    np.testing.assert_array_equal(licensed_image,before)
    again,_ = render_skin(licensed_image,licensed_landmarks,plan(.9),learned_proposal=proposal)
    np.testing.assert_array_equal(output,again)


def test_learned_equal_proposal_or_zero_age_is_exact_identity(licensed_image,licensed_landmarks):
    for scores,proposal in ((plan(1),licensed_image.copy()),
                            (plan(0),255-licensed_image)):
        output,metrics = render_skin(licensed_image,licensed_landmarks,scores,learned_proposal=proposal)
        np.testing.assert_array_equal(output,licensed_image)
        assert metrics['applied_features'] == []
        assert metrics['changed_pixels'] == 0


def test_learned_mask_and_proposal_validation(licensed_image,licensed_landmarks):
    with pytest.raises(ValueError):
        render_skin(licensed_image,licensed_landmarks,plan(1),learned_proposal=np.zeros((5,5,3),np.uint8))
    proposal = 255-licensed_image
    output,metrics = render_skin(licensed_image,licensed_landmarks,plan(1),learned_proposal=proposal,
                                skin_mask=np.zeros(licensed_image.shape[:2],np.float32))
    np.testing.assert_array_equal(output,licensed_image)
    assert metrics['applied_features'] == []


def test_very_small_learned_strength_does_not_empty_tone_statistics(licensed_image,licensed_landmarks):
    output,metrics = render_skin(licensed_image,licensed_landmarks,plan(.001),
                                learned_proposal=255-licensed_image)
    assert output.shape == licensed_image.shape
    assert metrics['changed_pixels'] == 0


def test_height_furrow_has_tissue_rolls_and_spatial_normal_relief():
    from experiments.geometry_renderer.skin_layers import _furrow_height,_relight_height
    path = np.column_stack((np.linspace(10,85,80),np.full(80,48.0)))
    height = _furrow_height((96,96),[path],300,depth=1,width=2)
    assert height[48,48] < -.5
    # A raised roll exists beside the depression, not just a dark RGB stroke.
    assert height.max() > .02
    source = np.full((96,96,3),180,np.float32)
    relit,metadata = _relight_height(source,height,np.ones((96,96),np.float32),300)
    assert (relit > source+.5).any()
    assert (relit < source-.5).any()
    assert metadata['height_range_px'][0] < 0 < metadata['height_range_px'][1]
    np.testing.assert_allclose(relit[:20],source[:20],atol=.05)
    flat,_ = _relight_height(source,np.zeros((96,96),np.float32),np.ones((96,96),np.float32),300)
    np.testing.assert_array_equal(flat,source)


def test_diagnostic_relief_uses_source_lighting_and_short_glabellar_paths(licensed_landmarks):
    from experiments.geometry_renderer.skin_layers import _furrow_height,_relight_height,_wrinkle_paths
    y,x = np.indices((96,96))
    left = np.repeat((230-x*1.6)[...,None],3,axis=2).astype(np.float32)
    right = np.repeat((70+x*1.6)[...,None],3,axis=2).astype(np.float32)
    path = np.column_stack((np.full(80,48.0),np.linspace(10,85,80)))
    height = _furrow_height((96,96),[path],300,depth=1,width=2)
    _,a = _relight_height(left,height,np.ones((96,96),np.float32),300)
    _,b = _relight_height(right,height,np.ones((96,96),np.float32),300)
    assert a['light_direction'][0] < b['light_direction'][0]
    fh = np.ptp(licensed_landmarks[:,1])
    for path in _wrinkle_paths(licensed_landmarks)['glabellar_lines']:
        assert np.ptp(path[:,1]) < .1*fh
