"""Age-plan behavior and provenance tests; no images, services or network."""
import csv
import hashlib
from pathlib import Path
import pytest
from experiments.geometry_renderer.age_plan import build_age_plan

ROOT = Path(__file__).resolve().parents[3]
REQUIRED = {'forehead_wrinkles','crow_feet','glabellar_lines','nasolabial_folds',
    'marionette_lines','perioral_lines','under_eye_bags','tear_trough','cheek_volume_loss',
    'skin_laxity','age_spots','skin_texture','hair_graying','neck_lines'}


def test_zero_age_delta_preserves_original_even_with_lifestyle():
    plan = build_age_plan(75,75,lifestyle={'smoking':1,'uv_exposure':1})
    assert plan['mode']=='illustrative_local'
    assert REQUIRED <= set(plan['features'])
    for feature in plan['features'].values():
        assert feature['baseline_current']==feature['baseline_target']
        assert feature['baseline_delta']==0 and feature['score']==0


def test_visible_elderly_demo_without_lifestyle_exposures():
    neutral = build_age_plan(30,75)
    zeros = build_age_plan(30,75,lifestyle={key:0 for key in ('smoking','uv_exposure','alcohol','stress','diet')})
    for name in REQUIRED:
        assert .6 <= neutral['features'][name]['score'] <= 1
        assert neutral['features'][name]['score']==zeros['features'][name]['score']
    assert neutral['metadata']['clinically_calibrated'] is False
    assert all('provisional' in feature['evidence_status'] for feature in neutral['features'].values())


def test_age_baselines_and_delta_are_monotonic_and_continuous():
    plans = [build_age_plan(30,target) for target in range(30,101)]
    for name in REQUIRED:
        previous = 0
        for plan in plans:
            feature=plan['features'][name]
            assert feature['baseline_target'] >= feature['baseline_current']
            assert previous <= feature['score'] <= 1
            assert feature['score']-previous < .06
            previous=feature['score']


def test_lifestyle_modifies_only_relevant_existing_graph_paths():
    neutral=build_age_plan(30,75)
    exposed=build_age_plan(30,75,lifestyle={'uv_exposure':1})
    assert exposed['features']['age_spots']['score'] > neutral['features']['age_spots']['score']
    assert exposed['features']['forehead_wrinkles']['score'] > neutral['features']['forehead_wrinkles']['score']
    # The source graph has no gray-hair pathway; no lifestyle claim is invented.
    assert exposed['features']['hair_graying']['score']==neutral['features']['hair_graying']['score']
    for name in REQUIRED:
        assert exposed['features'][name]['baseline_delta']==neutral['features'][name]['baseline_delta']


def test_missing_features_have_no_borrowed_citations():
    plan=build_age_plan(30,75)
    for name in ('hair_graying','glabellar_lines','perioral_lines','under_eye_bags',
                 'tear_trough','cheek_volume_loss','neck_lines'):
        feature=plan['features'][name]
        assert feature['evidence_status']=='provisional_missing_feature_overlay'
        assert feature['citations']==[] and feature['graph_paths']==[]
        assert feature['graph_node_ids']==[]
    assert 'under_eye_circles' not in plan['features']['under_eye_bags']['graph_node_ids']


def test_real_graph_paths_and_citations_are_exact_source_records():
    plan=build_age_plan(30,75)
    edges=list(csv.DictReader((ROOT/'data/edges_v03.csv').open(encoding='utf-8-sig')))
    records={(e['source'],e['target'],e['relation'],e['evidence']) for e in edges}
    for feature in plan['features'].values():
        for path in feature['graph_paths']:
            assert path['nodes'][-1] in feature['graph_node_ids']
            for step in path['steps']:
                assert (step['source'],step['target'],step['relation'],step['evidence']) in records
    forehead=plan['features']['forehead_wrinkles']
    assert forehead['graph_support_status']=='incomplete_graph_paths'
    assert 'Mechanistic mapping' not in forehead['citations']
    assert any(step['missing_citation'] for path in forehead['graph_paths'] for step in path['steps'])
    # Existing women→marionette age_max40 evidence is not reapplied at age75.
    assert plan['features']['marionette_lines']['graph_paths']==[]
    assert plan['features']['marionette_lines']['score'] > 0


def test_graph_hashes_identify_exact_bytes_and_override_accepts_data_directory():
    plan=build_age_plan(30,75,graph_root=ROOT/'data')
    graph=plan['metadata']['source_graph']
    assert graph['version']=='v03' and graph['node_count']==80 and graph['edge_count']==98
    for kind in ('nodes','edges'):
        path=ROOT/'data'/f'{kind}_v03.csv'
        assert graph['files'][kind]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('ages',[(17,75),(30,101),(75,30),(True,75),(30,75.5)])
def test_bad_ages_are_rejected(ages):
    with pytest.raises(ValueError):
        build_age_plan(*ages)


@pytest.mark.parametrize('lifestyle',[[],{'unknown':.5},{'smoking':-1},{'smoking':float('nan')},
                                    {'smoking':float('inf')},{'smoking':True},{'smoking':'1'}])
def test_bad_lifestyle_is_rejected(lifestyle):
    with pytest.raises(ValueError):
        build_age_plan(30,75,lifestyle=lifestyle)


def test_overlay_graph_has_explicit_namespaced_baseline_paths_for_all_features():
    from experiments.geometry_renderer.age_plan import FEATURES, baseline_overlay_graph
    overlay=baseline_overlay_graph()
    nodes={node['id']:node for node in overlay['nodes']}
    assert len(nodes)==len(FEATURES)+2
    assert len(overlay['edges'])==len(FEATURES)+1
    assert all(node_id.startswith('local.') for node_id in nodes)
    assert nodes['local.intrinsic_aging']['source_graph_node_ids']==['intrinsic_aging']
    assert nodes['local.demo_normal_aging_baseline']['layer']=='demo_render_controller'
    edges={(edge['source'],edge['target']):edge for edge in overlay['edges']}
    assert ('local.intrinsic_aging','local.demo_normal_aging_baseline') in edges
    for feature in FEATURES:
        assert nodes[f'local.{feature}']['feature_id']==feature
        assert ('local.demo_normal_aging_baseline',f'local.{feature}') in edges
    assert overlay['causal_claim'] is False and overlay['clinically_calibrated'] is False
    for edge in overlay['edges']:
        assert edge['source'] in nodes and edge['target'] in nodes
        assert edge['relation']=='provisional_render_mapping'
        assert edge['evidence_status']=='provisional_render_mapping'
        assert edge['clinically_calibrated'] is False and edge['causal_claim'] is False
        assert edge['calibration_status']=='uncalibrated' and edge['citations']==[]


def test_plan_overlay_uses_same_curve_constants_and_never_rewrites_source_graph():
    from experiments.geometry_renderer.age_plan import baseline_overlay_graph
    source_paths=[ROOT/'data/nodes_v03.csv',ROOT/'data/edges_v03.csv']
    original={path:path.read_bytes() for path in source_paths}
    plan=build_age_plan(30,75)
    overlay_nodes={node['id']:node for node in plan['overlay_graph']['nodes']}
    for feature_id,feature in plan['features'].items():
        node=overlay_nodes[feature['overlay_node_id']]
        assert node['feature_id']==feature_id
        assert node['demo_curve']==feature['demo_curve']
        assert node['render_channel']==feature['render_channel']
        assert node['citations']==[]  # Source references remain separate from authored overlay edges.
    # The exported overlay is JSON-serializable for UI/download inspection.
    import json
    assert json.loads(json.dumps(plan['overlay_graph']))==plan['overlay_graph']
    # Every export is a new object; callers cannot mutate global renderer specs.
    plan['overlay_graph']['nodes'][0]['source_graph_node_ids'].append('fabricated')
    assert baseline_overlay_graph()['nodes'][0]['source_graph_node_ids']==['intrinsic_aging']
    for path,content in original.items():
        assert path.read_bytes()==content
