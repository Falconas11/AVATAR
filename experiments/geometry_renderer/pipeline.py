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
                       learned_proposal: np.ndarray | None = None) -> AgingResult:
    start = perf_counter()
    enabled = set(enabled_layers)
    if enabled-{'geometry','skin','hair'}:
        raise ValueError('Unknown aging render layer')
    geometry = {'nose': .65, 'cheeks': .65, 'jowls': .65} | (geometry_strengths or {})
    if set(geometry) != {'nose','cheeks','jowls'}:
        raise ValueError('Unknown geometry control')
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in [skin_strength,hair_strength,*geometry.values()]):
        raise ValueError('Layer strengths must be finite fractions between 0 and 1')
    plan = build_age_plan(current_age,target_age,lifestyle=lifestyle)
    features = plan['features']
    for feature in features.values():
        feature['age_score'] = feature['score']
        feature['render_operator_enabled'] = False
    scores = {name: record['score'] for name,record in features.items()}
    delta = displacement_field(prepared.points,prepared.face_count,current_age=current_age,target_age=target_age,
        nose=scores['nasal_tip_ptosis']*geometry['nose'] if 'geometry' in enabled else 0,
        cheeks=scores['cheek_descent']*geometry['cheeks'] if 'geometry' in enabled else 0,
        jowls=scores['jowls']*geometry['jowls'] if 'geometry' in enabled else 0, progression=1)
    destination, safety = safe_destination(prepared,delta)
    h,w = prepared.image.shape[:2]
    warped_regions = {}
    if learned_proposal is not None and (learned_proposal.shape != prepared.image.shape or learned_proposal.dtype != np.uint8):
        raise ValueError('Learned appearance proposal must match the RGB prepared image')
    warped_proposal = learned_proposal
    stage_start = perf_counter()
    if np.array_equal(destination,prepared.points):
        geometry_image = prepared.image.copy()
        warped_regions = {k: regions[k] for k in ('skin','neck','hair') if regions and k in regions}
    else:
        x,y = inverse_maps(prepared.points,destination,prepared.triangles,w,h)
        geometry_image = cv2.remap(prepared.image,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        if learned_proposal is not None:
            warped_proposal = cv2.remap(learned_proposal,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        for key in ('skin','neck','hair'):
            if regions and key in regions:
                mask = np.asarray(regions[key],dtype=np.float32)
                if mask.shape != (h,w) or not np.isfinite(mask).all():
                    raise ValueError('Semantic masks must match the finite prepared image')
                warped_regions[key] = cv2.remap(mask,x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    geometry_ms = (perf_counter()-stage_start)*1000
    stage_start = perf_counter()
    # The FRAN proposal is already conditioned on source and target ages. Use
    # the plan as an age gate and relative regional control, not a second global
    # age fade that would turn a target-75 proposal back toward young skin.
    appearance_score = max((record['score'] for record in features.values()
        if record['render_channel'] in ('skin','wrinkle')),default=0)
    normalizer = appearance_score if learned_proposal is not None and appearance_score>0 else 1
    skin_plan = {**plan,'features':{name:{**record,'score':min(1.,record['score']/normalizer)*skin_strength}
        for name,record in features.items()}}
    if 'skin' in enabled:
        skin_image,skin_metrics = render_skin(geometry_image,destination,skin_plan,
            skin_mask=warped_regions.get('skin'),neck_mask=warped_regions.get('neck'),
            learned_proposal=warped_proposal)
    else:
        skin_image,skin_metrics = geometry_image.copy(),{'enabled':False}
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
    metrics = {'geometry_ms':geometry_ms,'skin_ms':skin_ms,'hair_ms':hair_ms,
        'total_ms':(perf_counter()-start)*1000,'geometry_changed_pixels':geometry_changed,
        'skin_changed_pixels':skin_changed,'hair_changed_pixels':hair_changed,
        'hair_mask_pixels':int((hair_mask>.5).sum()),'skin':skin_metrics,'hair':hair_metrics,
        'enabled_layers':sorted(enabled),'segmentation_backend':regions.get('backend','none') if regions else 'none',
        'learned_skin_used':bool(learned_proposal is not None and 'skin' in enabled and skin_changed),
        'disclaimer':'Per-feature flags mean requested operators, not neural feature attribution. Pixel/layer diagnostics validate execution, not perceived age or biological prediction.'}
    return AgingResult(final,geometry_image,skin_image,destination,safety,plan,metrics)
