"""Local age baseline → geometry → skin relief/texture → segmented hair.

Scores and rendering operators are explicit visual hypotheses. Graph reference
paths inform the vocabulary/relative modifiers, not calibrated age predictions.
No services, photo writes, photo logging, or global image caches are used here.
"""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
from .age_plan import build_age_plan
from .core import PreparedFace, displacement_field, inverse_maps, safe_destination
from .skin_layers import render_skin
from .hair_layers import gray_hair
from .mechanism_layers import volume_displacement,render_surface,ear_warp


@dataclass(frozen=True)
class AgingResult:
    image: np.ndarray
    geometry_image: np.ndarray
    skin_image: np.ndarray
    points: np.ndarray
    safety_scale: float
    plan: dict
    metrics: dict


def render_local_aging(prepared: PreparedFace, regions: dict | None, current_age: int,
                       target_age: int, *, lifestyle: dict | None = None,
                       geometry_strengths: dict | None = None, skin_strength: float = 1,
                       hair_strength: float = 1, enabled_layers=('geometry','skin','hair'),
                       learned_proposal: np.ndarray | None = None, mechanism_strengths: dict | None = None,
                       feature_strengths: dict | None = None) -> AgingResult:
    start = perf_counter()
    enabled = set(enabled_layers)
    if enabled-{'geometry','skin','hair'}:
        raise ValueError('Unknown aging render layer')
    geometry = {'nose': .65, 'cheeks': .65, 'jowls': .65} | (geometry_strengths or {})
    if set(geometry) != {'nose','cheeks','jowls'}:
        raise ValueError('Unknown geometry control')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in [skin_strength,hair_strength,*geometry.values()]):
        raise ValueError('Layer strengths must be finite fractions between 0 and 1')
    plan = build_age_plan(current_age,target_age,lifestyle=lifestyle,mechanism_strengths=mechanism_strengths,feature_strengths=feature_strengths)
    features = plan['features']
    for feature in features.values():
        feature['age_score'] = feature['baseline_score']
        feature['render_operator_enabled'] = False
    scores = {name: record['score'] for name,record in features.items()}
    delta = displacement_field(prepared.points,prepared.face_count,current_age=current_age,target_age=target_age,
        nose=max(scores['nasal_tip_ptosis'],scores['increased_nasal_length']*.7)*geometry['nose'] if 'geometry' in enabled else 0,
        cheeks=scores['cheek_descent']*geometry['cheeks'] if 'geometry' in enabled else 0,
        jowls=scores['jowls']*geometry['jowls'] if 'geometry' in enabled else 0, progression=1)
    if 'geometry' in enabled:
        delta += volume_displacement(prepared.points,prepared.face_count,plan)*max(geometry.values())
    destination, safety = safe_destination(prepared,delta)
    h,w = prepared.image.shape[:2]
    warped_regions = {}
    if learned_proposal is not None and (learned_proposal.shape != prepared.image.shape or learned_proposal.dtype != np.uint8):
        raise ValueError('Learned appearance proposal must match the RGB prepared image')
    warped_proposal = learned_proposal
    stage_start = perf_counter()
    if np.array_equal(destination,prepared.points):
        geometry_image = prepared.image.copy()
        warped_regions = {k: regions[k] for k in ('skin','neck','hair','ears') if regions and k in regions}
    else:
        x,y = inverse_maps(prepared.points,destination,prepared.triangles,w,h)
        geometry_image = cv2.remap(prepared.image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        if learned_proposal is not None:
            warped_proposal = cv2.remap(learned_proposal,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        for key in ('skin','neck','hair','ears'):
            if regions and key in regions:
                mask = np.asarray(regions[key],dtype=np.float32)
                if mask.shape != (h,w) or not np.isfinite(mask).all():
                    raise ValueError('Semantic masks must match the finite prepared image')
                warped_regions[key] = cv2.remap(mask,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    ear_metrics={'available':False,'changed_pixels':0,'strength':0}
    if 'geometry' in enabled:
        before_ears=geometry_image
        geometry_image,new_regions,ear_metrics=ear_warp(geometry_image,warped_regions,plan)
        if warped_proposal is not None:
            warped_proposal,_,_=ear_warp(warped_proposal,warped_regions,plan)
        warped_regions=new_regions
    geometry_ms = (perf_counter()-stage_start)*1000
    stage_start = perf_counter()
    # The FRAN proposal is already conditioned on source and target ages. Use
    # the plan as an age gate and relative regional control, not a second global
    # age fade that would turn a target-75 proposal back toward young skin.
    appearance_score = max((record['baseline_score'] for record in features.values()
        if record['render_channel'] in ('skin','wrinkle')),default=0)
    normalizer = appearance_score if learned_proposal is not None and appearance_score>0 else 1
    skin_plan = {**plan,'features':{name:{**record,'score':min(1.,record['score']/normalizer)*skin_strength}
        for name,record in features.items()}}
    if 'skin' in enabled:
        learned_plan = {**skin_plan,'features':{name:{**record,'score':min(1.,record['baseline_score']/normalizer)*skin_strength}
            for name,record in features.items()}} if learned_proposal is not None else skin_plan
        skin_image,skin_metrics = render_skin(geometry_image,destination,learned_plan,
            skin_mask=warped_regions.get('skin'),neck_mask=warped_regions.get('neck'),
            learned_proposal=warped_proposal)
    else:
        skin_image,skin_metrics = geometry_image.copy(),{'enabled':False}
    surface_metrics={'operators':{}}
    if 'skin' in enabled:
        skin_image,surface_metrics=render_surface(skin_image,destination,skin_plan,warped_regions.get('skin'),
            learned_detail=learned_proposal is not None)
    skin_ms = (perf_counter()-stage_start)*1000
    stage_start = perf_counter()
    hair_mask = warped_regions.get('hair',np.zeros((h,w),np.float32))
    if hair_mask.shape != (h,w) or not np.isfinite(hair_mask).all():
        raise ValueError('Hair mask must match the finite prepared image')
    if 'hair' in enabled:
        final,hair_metrics = gray_hair(skin_image,hair_mask,scores['hair_graying']*hair_strength)
    else:
        final,hair_metrics = skin_image.copy(),{'enabled':False}
    hair_ms = (perf_counter()-stage_start)*1000
    skin_changed = int(np.any(skin_image!=geometry_image,axis=2).sum())
    hair_changed = int(np.any(final!=skin_image,axis=2).sum())
    geometry_changed = int(np.any(geometry_image!=prepared.image,axis=2).sum())
    for name in ('nasal_tip_ptosis','cheek_descent','jowls'):
        features[name]['render_operator_enabled'] = bool('geometry' in enabled and scores[name]>0 and geometry_changed)
    skin_ids = {'forehead_wrinkles','crow_feet','glabellar_lines','nasolabial_folds','marionette_lines',
        'perioral_lines','under_eye_bags','tear_trough','cheek_volume_loss','skin_laxity',
        'age_spots','skin_texture','neck_lines'}
    for name in skin_ids:
        features[name]['render_operator_enabled'] = bool('skin' in enabled and scores[name]>0 and skin_strength>0 and skin_changed)
    # Distinguish unavailable neck coverage from a merely planned neck feature.
    if not warped_regions.get('neck',np.array([])).any():
        features['neck_lines']['render_operator_enabled'] = False
    features['hair_graying']['render_operator_enabled'] = bool(hair_changed)
    # Complete source-KG coverage is separate from actual visibility. Neural
    # residual requests are regional controls, never per-feature causal traces.
    coverage={}
    geometry_ids={'nasal_tip_ptosis','increased_nasal_length','lower_face_sagging','jowls','facial_puffiness'}
    ear_ids={'earlobe_elongation','earlobe_ptosis'}
    wrinkle_ids={'forehead_wrinkles','facial_wrinkles','crow_feet','nasolabial_folds','deep_nasolabial_folds','marionette_lines'}
    for source_id,mapped in plan['source_feature_coverage'].items():
        channel='geometry' if source_id in geometry_ids|ear_ids else 'skin'
        operator=('semantic_ear_warp' if source_id in ear_ids else 'mesh_geometry' if source_id in geometry_ids
            else 'regional_learned_skin_residual' if source_id in wrinkle_ids else 'explicit_surface_proxy')
        available=bool(ear_metrics['available']) if source_id in ear_ids else bool(learned_proposal is not None) if source_id in wrinkle_ids else True
        coverage[source_id]={'operator':operator,'available':available,'enabled':channel in enabled,
            'intensity':max(scores[name] for name in mapped),'source_feature_ids':mapped,
            'status':'unobservable_no_confident_ear_region' if not available and source_id in ear_ids else
                'coarse_surface_proxy_not_nodules' if source_id=='nodular_elastosis' else 'implemented_visual_proxy',
            'biological_simulation':False,'individual_neural_feature_verified':False}
    for source_id,record in coverage.items():
        for mapped in record['source_feature_ids']:
            features[mapped]['render_operator_enabled'] = bool(record['enabled'] and record['available'] and record['intensity']>0)
    metrics = {'geometry_ms':geometry_ms,'skin_ms':skin_ms,'hair_ms':hair_ms,
        'total_ms':(perf_counter()-start)*1000,'geometry_changed_pixels':geometry_changed,
        'skin_changed_pixels':skin_changed,'hair_changed_pixels':hair_changed,
        'hair_mask_pixels':int((hair_mask>.5).sum()),'skin':skin_metrics,'hair':hair_metrics,
        'feature_coverage':coverage,'mechanisms':plan['mechanisms'],'ear_warp':ear_metrics,
        'lifestyle_scenarios':plan['lifestyle_scenarios'],
        'surface':surface_metrics,'enabled_layers':sorted(enabled),'segmentation_backend':regions.get('backend','none') if regions else 'none',
        'learned_skin_used':bool(learned_proposal is not None and 'skin' in enabled and skin_changed),
        'disclaimer':'Per-feature flags mean requested operators, not neural feature attribution. Pixel/layer diagnostics validate execution, not perceived age or biological prediction.'}
    return AgingResult(final,geometry_image,skin_image,destination,safety,plan,metrics)
