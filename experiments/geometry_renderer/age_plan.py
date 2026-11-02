"""Auditable, local-only age plans over the repository KG plus authored overlays.

The CSV graph supplies feature names and *reference paths*. It does not supply
calibrated age/severity curves, wrinkle depths, gray-hair percentages or pixel
operations. Every baseline below is therefore an explicit provisional visual
assumption. A score is an additional rendering intensity, never a probability.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from io import StringIO
import math
from numbers import Real
from pathlib import Path
from typing import Any
from .lifestyle import feature_contributions, scenario_evidence


@dataclass(frozen=True)
class FeatureSpec:
    region: str
    render_channel: str
    onset_age: int
    full_age: int
    visual_gain: float
    graph_ids: tuple[str, ...] = ()
    mapping: str = ''


# These ages and gains are authored demo controls, NOT empirical measurements.
# Curves intentionally make a 30→75 local demonstration perceptibly elderly.
FEATURES: dict[str, FeatureSpec] = {
    'forehead_wrinkles': FeatureSpec('forehead', 'wrinkle', 24, 84, .90, ('forehead_wrinkles',)),
    'crow_feet': FeatureSpec('outer eye corners', 'wrinkle', 22, 84, .90, ('crow_feet',)),
    'glabellar_lines': FeatureSpec('between eyebrows', 'wrinkle', 28, 86, .85),
    'nasolabial_folds': FeatureSpec('nose-to-mouth folds', 'wrinkle', 20, 85, .90,
        ('nasolabial_folds', 'deep_nasolabial_folds'), 'Explicit display grouping of the graph\'s two nasolabial-fold nodes.'),
    'marionette_lines': FeatureSpec('mouth corners to chin', 'wrinkle', 30, 86, .90, ('marionette_lines',)),
    'perioral_lines': FeatureSpec('skin around lips', 'wrinkle', 30, 88, .85),
    'under_eye_bags': FeatureSpec('lower eyelids', 'skin', 30, 88, .80,
        mapping='An authored bag/shadow overlay; under-eye circles are not evidence for eye-bag anatomy.'),
    'tear_trough': FeatureSpec('infraorbital hollow', 'skin', 24, 86, .80),
    'cheek_volume_loss': FeatureSpec('midface and cheeks', 'geometry', 22, 88, .85),
    'skin_laxity': FeatureSpec('lower face and jawline', 'geometry', 25, 85, .90,
        ('lower_face_sagging',), 'Display alias limited to lower-face sagging, not all-body skin elasticity.'),
    'age_spots': FeatureSpec('cheeks and forehead', 'skin', 30, 88, .80,
        ('sunspots',), 'Age-spots display cue is mapped to the graph\'s sunspots node; age incidence is not inferred from it.'),
    'skin_texture': FeatureSpec('facial skin', 'skin', 24, 86, .85,
        ('dry_skin', 'enlarged_pores'), 'Combines graph dryness/coarseness and pore cues as an illustrative skin channel.'),
    'hair_graying': FeatureSpec('existing hair region', 'hair', 28, 82, .90,
        mapping='No gray-hair node exists in v03. Gray-hair rendering is a provisional local overlay, not a clinical percentage.'),
    'neck_lines': FeatureSpec('visible neck skin', 'wrinkle', 28, 88, .90),
    'nasal_tip_ptosis': FeatureSpec('nose tip', 'geometry', 30, 90, .80, ('nasal_tip_ptosis',)),
    'cheek_descent': FeatureSpec('midface and cheeks', 'geometry', 25, 85, .85),
    'jowls': FeatureSpec('lower cheek and jawline', 'geometry', 30, 88, .90, ('jowls',)),
}

# Source-KG feature vocabulary is covered explicitly. Closely related aliases
# share operators via max aggregation, never summed duplicate severities.
FEATURES.update({
 'facial_wrinkles': FeatureSpec('facial skin','wrinkle',24,86,.9,('facial_wrinkles',)),
 'deep_nasolabial_folds': FeatureSpec('nasolabial region','wrinkle',25,88,.9,('deep_nasolabial_folds',)),
 'dry_skin': FeatureSpec('facial skin','skin',28,88,.65,('dry_skin',)),
 'dull_skin_tone': FeatureSpec('facial skin','skin',30,90,.45,('dull_skin_tone',)),
 'facial_puffiness': FeatureSpec('cheeks','geometry',40,95,.22,('facial_puffiness',)),
 'lower_face_sagging': FeatureSpec('lower face','geometry',25,85,.9,('lower_face_sagging',)),
 'under_eye_circles': FeatureSpec('under eyes','skin',25,88,.6,('under_eye_circles',)),
 'uneven_skin_tone': FeatureSpec('facial skin','skin',30,90,.5,('uneven_skin_tone',)),
 'hyperpigmentation': FeatureSpec('cheeks','skin',35,95,.3,('hyperpigmentation',)),
 'inconsistent_facial_pigmentation': FeatureSpec('facial skin','skin',35,95,.3,('inconsistent_facial_pigmentation',)),
 'sunspots': FeatureSpec('cheeks and forehead','skin',30,88,.65,('sunspots',)),
 'nodular_elastosis': FeatureSpec('photoexposed skin','skin',45,95,.3,('nodular_elastosis',),
   'Renders only a coarse photoaged-surface proxy; does not synthesize or diagnose pathological nodules.'),
 'enlarged_pores': FeatureSpec('nose and medial cheeks','skin',25,85,.45,('enlarged_pores',)),
 'broken_capillaries': FeatureSpec('nose and cheeks','skin',40,95,.16,('broken_capillaries',)),
 'facial_redness': FeatureSpec('nose and cheeks','skin',40,95,.16,('facial_redness',)),
 'spider_veins': FeatureSpec('cheeks','skin',45,95,.14,('spider_veins',)),
 'increased_nasal_length': FeatureSpec('nose','geometry',30,90,.55,('increased_nasal_length',)),
 'earlobe_elongation': FeatureSpec('visible earlobes','geometry',35,90,.7,('earlobe_elongation',)),
 'earlobe_ptosis': FeatureSpec('visible earlobes','geometry',35,90,.7,('earlobe_ptosis',)),
})

LIFESTYLE_FACTORS = frozenset({'smoking', 'uv_exposure', 'alcohol', 'stress', 'diet'})
_PLACEHOLDERS = frozenset({'', 'n/a', 'na', 'none', 'unknown', 'tbd', 'mechanistic mapping'})


def _curve_spec(spec: FeatureSpec) -> dict[str, int | float]:
    # One source of curve constants for renderer plans and graph exports.
    return {'onset_age': spec.onset_age, 'full_age': spec.full_age, 'visual_gain': spec.visual_gain}


def baseline_overlay_graph() -> dict[str, Any]:
    """Export an explicit local controller graph, without altering the source KG.

    Every node is namespaced so local rendering assumptions cannot be mistaken
    for original v03 physiological knowledge. The source age concept is only a
    vocabulary alias: the overlay uses current/target ages, not a claim that the
    graph's numeric intrinsic-aging activation calibrated these curves.
    """
    age_id = 'local.intrinsic_aging'
    baseline_id = 'local.demo_normal_aging_baseline'
    nodes: list[dict[str, Any]] = [
        {'id': age_id, 'label': 'Age input (local alias of intrinsic aging)',
         'layer': 'source_concept_alias', 'source_graph_node_ids': ['intrinsic_aging'],
         'inputs': ['current_age', 'target_age'], 'clinically_calibrated': False,
         'evidence_status': 'provisional_render_mapping', 'citations': []},
        {'id': baseline_id, 'label': 'Authored normal-aging demonstration baseline',
         'layer': 'demo_render_controller', 'source_graph_node_ids': [],
         'clinically_calibrated': False, 'evidence_status': 'provisional_render_mapping',
         'citations': [], 'curve_family': 'smoothstep',
         'output': 'additional rendering intensity from positive target-minus-current baseline delta'},
    ]
    edges: list[dict[str, Any]] = []

    def add_edge(source: str, target: str) -> None:
        edges.append({'id': f'local.edge.{len(edges)}', 'source': source, 'target': target,
            'relation': 'provisional_render_mapping',
            'evidence_status': 'provisional_render_mapping', 'clinically_calibrated': False,
            'calibration_status': 'uncalibrated', 'causal_claim': False, 'citations': [],
            'meaning': 'Authored demo control relationship; not a physiological causal edge or model-internal explanation.'})

    add_edge(age_id, baseline_id)
    for feature_id, spec in FEATURES.items():
        overlay_id = f'local.{feature_id}'
        nodes.append({'id': overlay_id, 'feature_id': feature_id,
            'label': feature_id.replace('_', ' '), 'layer': 'demo_render_feature',
            'region': spec.region, 'render_channel': spec.render_channel,
            'source_graph_node_ids': list(spec.graph_ids), 'demo_curve': _curve_spec(spec),
            'clinically_calibrated': False, 'evidence_status': 'provisional_render_mapping',
            'citations': [], 'mapping_assumption': spec.mapping or
                'The local age curve is authored even when a source graph feature has the same name.'})
        add_edge(baseline_id, overlay_id)
    return {'id': 'local.age_baseline_overlay.v0.2', 'namespace': 'local.',
        'graph_type': 'demo_only_render_controller_overlay', 'source_graph_version': 'v03',
        'clinically_calibrated': False, 'causal_claim': False, 'citations': [],
        'nodes': nodes, 'edges': edges,
        'disclaimer': 'Separate provisional rendering overlay. New edges are not added to the source knowledge graph, are not physiological causal claims, and do not explain learned model internals.'}


def _age_baseline(age: int, spec: FeatureSpec) -> float:
    """Bounded monotonic C1 smoothstep, before/onset=0 and after/full=1."""
    t = min(1.0, max(0.0, (age-spec.onset_age)/(spec.full_age-spec.onset_age)))
    return t*t*(3-2*t)


def _reference_status(evidence: str, notes: str = '') -> tuple[bool, bool]:
    text = f'{evidence} {notes}'.lower()
    provisional = any(token in text for token in ('provisional', 'hypothesis', 'unverified', 'citation needed'))
    usable_label = evidence.strip().lower() not in _PLACEHOLDERS and not _provisional_label(evidence)
    return usable_label, provisional or not usable_label


def _provisional_label(evidence: str) -> bool:
    return any(token in evidence.lower() for token in ('provisional', 'hypothesis', 'unverified', 'citation needed'))


def _read_graph(graph_root: Path | None) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    root = Path(graph_root) if graph_root is not None else Path(__file__).resolve().parents[2]
    data = root/'data' if (root/'data'/'nodes_v03.csv').is_file() else root
    paths = {'nodes': data/'nodes_v03.csv', 'edges': data/'edges_v03.csv'}
    rows: dict[str, list[dict[str, str]]] = {}
    files = {}
    for kind, path in paths.items():
        content = path.read_bytes()
        files[kind] = {'filename': path.name, 'sha256': hashlib.sha256(content).hexdigest()}
        rows[kind] = list(csv.DictReader(StringIO(content.decode('utf-8-sig'))))
    nodes = {row['id']: row for row in rows['nodes']}
    edges = rows['edges']
    if any(edge['source'] not in nodes or edge['target'] not in nodes for edge in edges):
        raise ValueError('Age-plan graph edges must reference existing graph nodes')
    metadata = {'version': 'v03', 'files': files, 'node_count': len(nodes), 'edge_count': len(edges),
                'role': 'Existing graph reference paths and feature vocabulary, not age-curve calibration.'}
    return nodes, edges, metadata


def _eligible(edge: dict[str, str], target_age: int) -> bool:
    # This API has no demographic input and does not import demographic priors
    # into an individual's face. Inapplicable source age ranges are not ignored.
    if edge.get('sex_condition') or edge.get('ethnicity_condition'):
        return False
    minimum, maximum = edge.get('age_min', ''), edge.get('age_max', '')
    return (not minimum or target_age >= float(minimum)) and (not maximum or target_age <= float(maximum))


def _graph_paths(graph_ids: tuple[str, ...], nodes: dict[str, dict[str, str]],
                 edges: list[dict[str, str]], target_age: int) -> list[dict[str, Any]]:
    incoming: dict[str, list[dict[str, str]]] = {}
    for edge in edges:
        if _eligible(edge, target_age):
            incoming.setdefault(edge['target'], []).append(edge)
    roots = LIFESTYLE_FACTORS | {'intrinsic_aging'}
    results: list[dict[str, Any]] = []

    def walk(node: str, seen: frozenset[str], remaining: int) -> list[list[dict[str, str]]]:
        if node in seen:
            return []
        if node in roots:
            return [[]]
        if not remaining:
            return []
        paths = []
        for edge in incoming.get(node, []):
            for prefix in walk(edge['source'], seen | {node}, remaining-1):
                paths.append(prefix+[edge])
                if len(paths) >= 12:
                    return paths
        return paths

    for node_id in graph_ids:
        if node_id not in nodes:
            continue
        for path in walk(node_id, frozenset(), 6):
            steps = []
            citations: list[str] = []
            for edge in path:
                evidence, notes = edge.get('evidence', '').strip(), edge.get('notes', '').strip()
                usable, provisional = _reference_status(evidence, notes)
                if usable and evidence not in citations:
                    citations.append(evidence)
                steps.append({'source': edge['source'], 'target': edge['target'], 'relation': edge['relation'],
                              'weight': float(edge.get('weight') or 1), 'evidence': evidence,
                              'notes': notes, 'provisional': provisional, 'missing_citation': not usable})
            results.append({'nodes': [path[0]['source']]+[edge['target'] for edge in path], 'steps': steps,
                'citations': citations,
                'evidence_status': 'incomplete_or_provisional' if any(s['provisional'] for s in steps) else 'unverified_graph_references',
                'path_kind': 'existing_graph_reference_not_render_trace',
                'disclaimer': 'Reference labels are copied from the repository and have not been independently verified. They do not calibrate the demo age curve.'})
    return results


def build_age_plan(current_age: int, target_age: int, *, lifestyle: dict | None = None,
                   graph_root: Path | None = None, mechanism_strengths: dict | None = None,
                   feature_strengths: dict | None = None) -> dict:
    """Return additional visible aging intensities without rendering or networking.

    Age baseline and lifestyle are deliberately separate: zero lifestyle factors
    still age, and equal ages ALWAYS return zero additional rendering. Lifestyle
    can only strengthen a positive baseline delta via applicable graph paths.
    """
    if (type(current_age) is not int or type(target_age) is not int or
            not 18 <= current_age <= target_age <= 100):
        raise ValueError('Use integer ages 18–100, with target age at least current age')
    if lifestyle is not None and not isinstance(lifestyle, dict):
        raise ValueError('lifestyle must be a dictionary or None')
    values = {}
    for factor, value in (lifestyle or {}).items():
        if factor not in LIFESTYLE_FACTORS:
            raise ValueError(f'Unsupported lifestyle factor: {factor}')
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Lifestyle factor values must be finite numbers between 0 and 1')
        values[factor] = float(value)
    nodes, edges, source_graph = _read_graph(graph_root)
    features: dict[str, dict[str, Any]] = {}
    for feature_id, spec in FEATURES.items():
        matched = tuple(node for node in spec.graph_ids if node in nodes)
        paths = _graph_paths(matched, nodes, edges, target_age)
        contributions = feature_contributions(feature_id, paths, values)
        modifier = 1+min(.35,sum(contributions.values()))
        current, target = _age_baseline(current_age, spec), _age_baseline(target_age, spec)
        delta = max(0.0, target-current)
        citations = list(dict.fromkeys(citation for path in paths for citation in path['citations']))
        missing = not matched
        assumptions = [
            'The smooth age curve, onset/full ages and visual gain are authored provisional demonstration parameters, not derived or clinically calibrated from graph citations.',
            'Score is additional rendering intensity for current→target, not disease risk, aging probability, millimeters, or percentage gray hair.',
            'Existing graph paths are reference context only; they do not explain a model\'s internals or establish individual causation.',
        ]
        if missing:
            assumptions.append('This feature is absent from the source graph; a local-only missing-feature overlay is used with no borrowed citations.')
        elif not paths:
            assumptions.append('A matching graph feature exists, but no applicable age/lifestyle reference path is available at the target age.')
        if spec.mapping:
            assumptions.append(spec.mapping)
        features[feature_id] = {'baseline_current': current, 'baseline_target': target,
            'baseline_delta': delta, 'score': min(1.0, delta*spec.visual_gain*modifier),
            'region': spec.region, 'render_channel': spec.render_channel,
            'overlay_node_id': f'local.{feature_id}',
            'evidence_status': 'provisional_missing_feature_overlay' if missing else 'provisional_age_baseline',
            'citations': citations, 'graph_paths': paths, 'graph_node_ids': list(matched),
            'graph_support_status': 'absent_from_graph' if missing else ('no_applicable_graph_path' if not paths else
                'incomplete_graph_paths' if any(p['evidence_status']=='incomplete_or_provisional' for p in paths) else 'unverified_graph_references'),
            'assumptions': assumptions, 'lifestyle_modifier': modifier,
            'lifestyle_contributions':contributions,
            'baseline_score':min(1.,delta*spec.visual_gain),
            'demo_curve': _curve_spec(spec)}
    mechanism_ids = {key for key,node in nodes.items() if node['layer']=='mechanism'}
    controls = dict(mechanism_strengths or {})
    feature_controls = dict(feature_strengths or {})
    for mapping,allowed in ((controls,mechanism_ids),(feature_controls,set(features))):
        if set(mapping)-allowed:
            raise ValueError('Unknown graph mechanism or render feature control')
        if any(isinstance(v,bool) or not isinstance(v,Real) or not math.isfinite(v) or not 0<=v<=1 for v in mapping.values()):
            raise ValueError('Mechanism/feature controls must be finite fractions')
    mechanisms = {}
    for mechanism_id in sorted(mechanism_ids):
        descendants={mechanism_id}
        for _ in range(len(nodes)):
            expanded=descendants | {e['target'] for e in edges if e['source'] in descendants}
            if expanded==descendants: break
            descendants=expanded
        affected=[key for key,f in features.items() if set(f['graph_node_ids']) & descendants]
        value=float(controls.get(mechanism_id,1))
        for key in affected:
            features[key]['score'] *= value
            features[key]['baseline_score'] *= value
            features[key].setdefault('mechanism_controls',[]).append(mechanism_id)
        mechanisms[mechanism_id]={'label':nodes[mechanism_id]['label'],
            'control':value,'affected_features':affected,'source_descendant_ids':sorted(descendants-{mechanism_id}),
            'implementation':'qualitative descendant-linked render controller',
            'biochemical_simulation':False,'clinically_calibrated':False}
    for key,f in features.items():
        f['score'] *= float(feature_controls.get(key,1))
        f['baseline_score'] *= float(feature_controls.get(key,1))
    source_features=[key for key,node in nodes.items() if node['layer']=='feature' and
        node['renderable'].lower()=='true' and node['aging_related'].lower()=='true']
    coverage={key:[name for name,f in features.items() if key in f['graph_node_ids']] for key in source_features}
    if any(not mappings for mappings in coverage.values()):
        raise ValueError('Source graph contains an aging feature without a render-plan mapping')
    return {'current_age': current_age, 'target_age': target_age, 'mode': 'illustrative_local',
        'features': features, 'lifestyle_scenarios':scenario_evidence(features,values), 'mechanisms':mechanisms,'source_feature_coverage':coverage,
        'overlay_graph': baseline_overlay_graph(),
        'metadata': {'source_graph': source_graph, 'baseline_model': 'local_demo_smoothstep_v0.2',
            'clinically_calibrated': False, 'lifestyle_values': values,
            'lifestyle_policy': 'Allowlisted visual routes require applicable source paths. Additive gains capped at35%; aliases are not summed. Diet is not rendered; no clinical doses, duration, interaction model or demographic priors.',
            'lifestyle_gain_limit': 1.35},
        'disclaimer': 'Illustrative local visual aging simulation. All age baselines and missing-feature overlays are provisional and uncalibrated; not a personalized or medical prediction.'}
