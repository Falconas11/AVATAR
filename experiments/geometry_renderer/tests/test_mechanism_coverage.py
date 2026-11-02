"""Coverage is checked against source CSVs, not a duplicated hard-coded list."""
import csv
from pathlib import Path
import numpy as np
import pytest
from experiments.geometry_renderer.age_plan import build_age_plan,FEATURES
from experiments.geometry_renderer.mechanism_layers import render_surface,ear_warp,volume_displacement
from experiments.geometry_renderer.skin_layers import render_skin
ROOT=Path(__file__).resolve().parents[3]


def test_every_source_aging_feature_and_mechanism_has_explicit_mapping():
    nodes=list(csv.DictReader((ROOT/'data/nodes_v03.csv').open(encoding='utf-8-sig')))
    source={n['id'] for n in nodes if n['layer']=='feature' and n['aging_related']=='True' and n['renderable']=='True'}
    mechanisms={n['id'] for n in nodes if n['layer']=='mechanism'}
    plan=build_age_plan(30,75)
    assert len(source)==25 and len(mechanisms)==9
    assert set(plan['source_feature_coverage'])==source
    assert set(plan['mechanisms'])==mechanisms
    assert all(m['affected_features'] for m in plan['mechanisms'].values())
    assert all(not m['biochemical_simulation'] for m in plan['mechanisms'].values())


@pytest.mark.parametrize('mechanism', ['oxidative_stress','mmp_activity','collagen_degradation',
 'elastic_fiber_damage','uv_dna_damage','pigmentation_dysregulation','skin_barrier_impairment',
 'nasal_support_weakening','auricular_tissue_laxity'])
def test_each_mechanism_can_disable_its_mapped_feature_group(mechanism):
    before=build_age_plan(30,75);after=build_age_plan(30,75,mechanism_strengths={mechanism:0})
    names=after['mechanisms'][mechanism]['affected_features']
    assert names and any(before['features'][name]['score']>0 for name in names)
    assert all(after['features'][name]['score']==0 for name in names)


def isolated(name):
    plan=build_age_plan(30,75)
    for key,f in plan['features'].items():f['score']=1. if key==name else 0.
    return plan


@pytest.mark.parametrize('name',['dry_skin','dull_skin_tone','under_eye_circles','sunspots',
 'hyperpigmentation','uneven_skin_tone','inconsistent_facial_pigmentation','enlarged_pores',
 'nodular_elastosis','broken_capillaries','facial_redness','spider_veins'])
def test_each_explicit_skin_feature_changes_masked_pixels(licensed_prepared,name):
    p=licensed_prepared
    result,metrics=render_surface(p.image,p.points,isolated(name))
    assert np.any(result!=p.image)
    np.testing.assert_array_equal(result[:10],p.image[:10])
    assert metrics['nodular_elastosis_representation']=='coarse_surface_proxy_not_nodules'


@pytest.mark.parametrize('name',['facial_wrinkles','deep_nasolabial_folds','crow_feet','forehead_wrinkles','marionette_lines'])
def test_learned_feature_regions_have_actual_output(licensed_prepared,name):
    p=licensed_prepared;proposal=np.maximum(p.image.astype(int)-18,0).astype('uint8')
    # A spatial illumination residual survives robust median matching.
    proposal[:,::3]=np.maximum(proposal[:,::3].astype(int)-25,0).astype('uint8')
    result,metrics=render_skin(p.image,p.points,isolated(name),learned_proposal=proposal)
    assert metrics['changed_pixels']>0
    assert not metrics['hand_authored_crease_paths_used']


def test_semantic_ear_deformation_and_missing_ears_are_distinct(synthetic):
    image=synthetic.image;mask=np.zeros(image.shape[:2],np.float32);mask[55:100,15:30]=1
    output,regions,metrics=ear_warp(image,{'ears':mask},isolated('earlobe_elongation'))
    assert metrics['available'] and metrics['changed_pixels']>0
    np.testing.assert_array_equal(output[0],image[0])
    original,_,missing=ear_warp(image,{},isolated('earlobe_ptosis'))
    assert not missing['available'];np.testing.assert_array_equal(original,image)


def test_midface_volume_and_puffiness_are_distinct_fields(synthetic):
    hollow=volume_displacement(synthetic.points,468,isolated('cheek_volume_loss'))
    puff=volume_displacement(synthetic.points,468,isolated('facial_puffiness'))
    assert np.any(hollow) and np.any(puff) and not np.allclose(hollow,puff)
    assert not hollow[468:].any() and not puff[468:].any()


def test_quality_first_canvas_has_nine_seam_free_windows():
    from experiments.geometry_renderer.conditional_skin import _windowed_residual
    class Constant:
        def setInput(self,value):self.value=value
        def forward(self):return np.ones((1,3,512,512),np.float32)*.1
    residual,count=_windowed_residual(Constant(),np.zeros((1024,1024,3),np.float32),30,75)
    assert count==9
    np.testing.assert_allclose(residual,.1,atol=1e-6)


def test_hair_confidence_is_not_mistaken_for_ungray_hair_fraction():
    from experiments.geometry_renderer.hair_layers import gray_hair
    image=np.full((32,32,3),[90,60,30],np.uint8)
    mask=np.full((32,32),.65,np.float32)
    result,_=gray_hair(image,mask,1.)
    assert np.mean(result[...,0].astype(float)-result[...,2]) < 30
