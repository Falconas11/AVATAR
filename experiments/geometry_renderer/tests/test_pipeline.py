"""Full local pipeline invariants; real models use only the licensed NASA fixture."""
from pathlib import Path
import numpy as np
import pytest
from experiments.geometry_renderer.pipeline import render_local_aging
from experiments.geometry_renderer.session import DemoSession,MAX_WORKING_SIDE
from .conftest import FIXTURE

@pytest.fixture(scope='module')
def prepared_session():
    session=DemoSession()
    session.load(FIXTURE.read_bytes(),'.png')
    session.ensure_regions()
    session.ensure_aged_proposal(30,75,quality='fast')
    return session


def test_zero_age_is_exact_identity_with_all_layers(prepared_session):
    s=prepared_session
    result=render_local_aging(s.prepared,s.regions,30,30,learned_proposal=s.aged_proposal)
    np.testing.assert_array_equal(result.image,s.prepared.image)
    assert all(f['score']==0 for f in result.plan['features'].values())
    assert result.metrics['skin_changed_pixels']==result.metrics['hair_changed_pixels']==0


def test_real_pipeline_uses_learned_detail_not_authored_crease_paths(prepared_session):
    s=prepared_session
    result=s.render_age(30,75)
    assert result.image.dtype==np.uint8 and result.image.shape==s.prepared.image.shape
    assert max(result.image.shape[:2]) <= MAX_WORKING_SIDE
    assert result.metrics['skin']['mode']=='learned_skin_appearance'
    assert not result.metrics['skin']['hand_authored_crease_paths_used']
    assert result.metrics['skin']['applied_features']==['learned_skin_appearance']
    assert result.metrics['skin_changed_pixels']>1000
    assert result.metrics['hair_changed_pixels']>1000
    assert not result.plan['metadata']['clinically_calibrated']
    assert result.plan['metadata']['source_graph']['version']=='v03'
    assert s.detection_count==s.segmentation_count==s.proposal_count==1


def test_layer_switches_and_all_off(prepared_session):
    s=prepared_session
    none=render_local_aging(s.prepared,s.regions,30,75,enabled_layers=(),learned_proposal=s.aged_proposal)
    np.testing.assert_array_equal(none.image,s.prepared.image)
    geometry=render_local_aging(s.prepared,s.regions,30,75,enabled_layers=('geometry',),learned_proposal=s.aged_proposal)
    np.testing.assert_array_equal(geometry.image,geometry.geometry_image)
    assert geometry.metrics['skin_changed_pixels']==geometry.metrics['hair_changed_pixels']==0
    assert not geometry.metrics['learned_skin_used']
    skin=render_local_aging(s.prepared,s.regions,30,75,enabled_layers=('skin',),learned_proposal=s.aged_proposal)
    np.testing.assert_array_equal(skin.geometry_image,s.prepared.image)
    assert skin.metrics['hair_changed_pixels']==0
    assert not np.array_equal(skin.image,s.prepared.image)


def test_inspection_cache_does_not_rerun_models(prepared_session):
    s=prepared_session
    first=s.render_age(30,75)
    render_count=s.render_count
    assert s.render_age(30,75) is first
    assert s.ensure_aged_proposal(30,75,quality='fast') is s.aged_proposal
    assert s.render_count==render_count
    assert s.detection_count==s.segmentation_count==s.proposal_count==1


def test_invalid_controls_rejected(synthetic):
    with pytest.raises(ValueError):
        render_local_aging(synthetic,None,30,75,skin_strength=2)
    with pytest.raises(ValueError):
        render_local_aging(synthetic,None,30,75,enabled_layers=('unknown',))


def test_session_invalidates_neural_cache_when_age_changes(licensed_landmarks):
    calls=[]
    s=DemoSession();s.load(FIXTURE.read_bytes(),'.png',detector=lambda image:licensed_landmarks)
    def generator(image,points,*,current_age,target_age,quality):
        calls.append((current_age,target_age));return image.copy(),{'test':True}
    first=s.ensure_aged_proposal(30,65,generator=generator)
    assert s.ensure_aged_proposal(30,65,generator=generator) is first
    s.ensure_aged_proposal(30,75,generator=generator)
    assert calls==[(30,65),(30,75)]
    assert s.detection_count==1
    s.clear()
    assert s.aged_proposal is s.regions is s.prepared is None
