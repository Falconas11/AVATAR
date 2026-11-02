"""Scenario routing and rendering invariants; generated fictional input only."""
from pathlib import Path
import numpy as np
import pytest
from experiments.geometry_renderer.age_plan import build_age_plan
from experiments.geometry_renderer.lifestyle import PRESETS, regional_gain
from experiments.geometry_renderer.pipeline import render_local_aging
from experiments.geometry_renderer.session import DemoSession


def test_routes_require_paths_and_do_not_infer_deficiency_or_gray_hair():
    base=build_age_plan(30,75)
    for factor in ('uv_exposure','smoking','alcohol','stress','diet'):
        plan=build_age_plan(30,75,lifestyle={factor:1})
        assert plan['features']['hair_graying']['score']==base['features']['hair_graying']['score']
        assert plan['features']['perioral_lines']['score']==base['features']['perioral_lines']['score']
        for name,f in plan['features'].items():
            assert f['baseline_score']==base['features'][name]['baseline_score']
            if f['score']>base['features'][name]['score']:
                assert any(p['nodes'][0]==factor for p in f['graph_paths'])
                assert factor in f['lifestyle_contributions']
        if factor=='diet':
            assert all(f['score']==base['features'][n]['score'] for n,f in plan['features'].items())
            assert not plan['lifestyle_scenarios']['diet']['features']


def test_combination_is_bounded_additive_and_preserves_missing_citations():
    a=build_age_plan(30,75,lifestyle={'uv_exposure':1})
    b=build_age_plan(30,75,lifestyle={'uv_exposure':1,'smoking':1})
    for n,f in b['features'].items():
        assert 1<=f['lifestyle_modifier']<=1.35
        assert f['score']>=a['features'][n]['score']
    paths=b['lifestyle_scenarios']['smoking']['features']['crow_feet']['paths']
    assert any(step['missing_citation'] for path in paths for step in path['steps'])
    assert any(p['evidence_status']=='incomplete_or_provisional' for p in paths)
    assert not b['lifestyle_scenarios']['smoking']['calibrated']


@pytest.fixture(scope='module')
def fictional_session():
    s=DemoSession()
    fixture=Path(__file__).parent/'fixtures/synthetic_portraits/S01.png'
    s.load(fixture.read_bytes(),'.png')
    # Semantic/model doubles keep the unit test offline and inexpensive. Real
    # model composition is exercised separately by validate_lifestyle.py.
    s.regions={k:np.ones(s.prepared.image.shape[:2],np.float32) for k in ('skin','hair')}
    y,x=np.indices(s.prepared.image.shape[:2])
    residual=(np.sin(x*.25)+np.cos(y*.18))*9
    proposal=np.clip(s.prepared.image.astype(float)+residual[...,None],0,255).astype(np.uint8)
    s.ensure_aged_proposal(30,75,generator=lambda *a,**k:(proposal,{}))
    return s


def test_scenarios_share_proposal_and_two_compositions_clear_together(fictional_session):
    s=fictional_session
    base=s.render_age(30,75)
    smoke=s.render_age(30,75,lifestyle={'smoking':1})
    count=s.render_count
    assert s.render_age(30,75) is base
    assert s.render_age(30,75,lifestyle={'smoking':1}) is smoke
    assert s.render_count==count and s.proposal_count==1
    assert not np.array_equal(smoke.image,base.image)
    assert all(f['age_score']==base.plan['features'][name]['age_score'] for name,f in smoke.plan['features'].items())
    assert len(s._render_cache)==2
    # New proposals invalidate both compositions, including quality changes.
    s.ensure_aged_proposal(30,75,quality='ultra',generator=lambda *a,**k:(s.aged_proposal.copy(),{}))
    assert not s._render_cache
    s.render_age(30,75)
    s.clear()
    assert s.prepared is None and s.aged_proposal is None and not s._render_cache


def test_regional_gain_does_not_disappear_in_normalization():
    s=DemoSession();s.load((Path(__file__).parent/'fixtures/synthetic_portraits/S02.png').read_bytes(),'.png')
    base=build_age_plan(30,75)
    uv=build_age_plan(30,75,lifestyle={'uv_exposure':1})
    smoke=build_age_plan(30,75,lifestyle={'smoking':1})
    a=regional_gain(s.prepared.image.shape,s.prepared.points,base['features'])
    b=regional_gain(s.prepared.image.shape,s.prepared.points,uv['features'])
    c=regional_gain(s.prepared.image.shape,s.prepared.points,smoke['features'])
    assert np.all(a==1) and 1.1<b.max()<=1.35 and 1.1<c.max()<=1.35
    assert not np.allclose(b,c)
    # Mouth emphasis is a documented smoking localization, not a missing
    # perioral graph node or gray-hair claim.
    mouth=np.rint((s.prepared.points[0]+s.prepared.points[17])/2).astype(int)
    assert c[mouth[1],mouth[0]]>b[mouth[1],mouth[0]]


@pytest.mark.parametrize('values',list(PRESETS.values())+[{'diet':1}])
def test_identity_and_disabled_layers_win_over_scenario(synthetic,values):
    proposal=255-synthetic.image
    equal=render_local_aging(synthetic,None,50,50,lifestyle=values,learned_proposal=proposal)
    off=render_local_aging(synthetic,None,30,75,lifestyle=values,enabled_layers=(),learned_proposal=proposal)
    np.testing.assert_array_equal(equal.image,synthetic.image)
    np.testing.assert_array_equal(off.image,synthetic.image)
