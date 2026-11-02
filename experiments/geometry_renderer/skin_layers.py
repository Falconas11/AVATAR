"""Deterministic, local wrinkle/skin illustration on a landmark-aligned skin mask.

The creases and shadows are authored visual cues, not simulated skin biology,
measured wrinkle depths, or a clinically calibrated age prediction. The module
never detects faces, downloads models, saves images, or logs image data.
"""
from __future__ import annotations
from .lifestyle import regional_gain

import cv2
import numpy as np

# MediaPipe's unrefined 468-point topology. These lists describe the external
# facial contour and *protected* features; they are not age-dependent geometry.
FACE_OVAL = (10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,
             377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109)
LEFT_EYE = (33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246)
RIGHT_EYE = (263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466)
LIPS = (61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185)
LEFT_BROW = (70,63,105,66,107,55,65,52,53,46)
RIGHT_BROW = (300,293,334,296,336,285,295,282,283,276)
SUPPORTED_FEATURES = ('forehead_wrinkles','crow_feet','glabellar_lines','nasolabial_folds',
    'marionette_lines','perioral_lines','under_eye_bags','tear_trough','cheek_volume_loss',
    'skin_laxity','age_spots','skin_texture')


def _points(points: np.ndarray) -> np.ndarray:
    value = np.asarray(points,dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 468 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError('Skin rendering requires 468 finite pixel landmarks')
    return value[:468]


def _score(plan: dict, feature_id: str) -> float:
    value = plan.get('features',{}).get(feature_id,{})
    value = value.get('score',0) if isinstance(value,dict) else value
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Skin scores must be finite values between zero and one')
    return float(value)


def skin_region_mask(image_shape, points: np.ndarray) -> np.ndarray:
    """Feather inward only; protected eyes, mouth and brows remain exact zero.

    The oval and holes constrain all layers, including blurred shading. Explicit
    masking is applied again after downsampling so interpolation cannot leak into
    protected features or the background.
    """
    p = _points(points)
    h,w = image_shape[:2]
    if min(h,w) < 2:
        raise ValueError('Skin rendering requires a nonempty image')
    face_width = max(float(np.ptp(p[:,0])),1)
    area = np.zeros((h,w),np.uint8)
    cv2.fillPoly(area,[np.rint(p[list(FACE_OVAL)]).astype(np.int32)],255)
    holes = np.zeros_like(area)
    for ids in (LEFT_EYE,RIGHT_EYE,LIPS,LEFT_BROW,RIGHT_BROW):
        hull = cv2.convexHull(np.rint(p[list(ids)]).astype(np.int32))
        cv2.fillConvexPoly(holes,hull,255)
    radius = max(1,int(round(face_width*.009)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    area[cv2.dilate(holes,kernel) > 0] = 0
    distance = cv2.distanceTransform(area,cv2.DIST_L2,5)
    feather = max(1.0,face_width*.018)
    return np.clip(distance/feather,0,1).astype(np.float32)


def _bezier(a,b,c,d=None,*,count=64) -> np.ndarray:
    t = np.linspace(0,1,count)[:,None]
    a,b,c = np.asarray(a),np.asarray(b),np.asarray(c)
    if d is None:
        return (1-t)**2*a + 2*(1-t)*t*b + t*t*c
    return (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*np.asarray(d)


def _gaussian(shape, center, radii) -> np.ndarray:
    y,x = np.ogrid[:shape[0],:shape[1]]
    return np.exp(-.5*(((x-center[0])/max(radii[0],1))**2+
                     ((y-center[1])/max(radii[1],1))**2)).astype(np.float32)


def _wrinkle_paths(p: np.ndarray) -> dict[str,list[np.ndarray]]:
    width,height = np.ptp(p,axis=0)
    paths = {name:[] for name in ('forehead_wrinkles','crow_feet','glabellar_lines',
        'nasolabial_folds','marionette_lines','perioral_lines','under_eye_bags','tear_trough')}
    # Forehead lines follow the forehead between brows and vertex10. A slight
    # curve and tapered endpoints avoid drawing a rectangular wrinkle grid.
    brow_y = float(p[[70,63,105,66,107,300,293,334,296,336],1].mean())
    center_x = float(p[9,0])
    top_y = float(p[10,1])
    outer_x = sorted((float(p[70,0]),float(p[300,0])))
    for i,fraction in enumerate((.30,.52,.74,.88)):
        y = brow_y-(brow_y-top_y)*fraction
        half = (outer_x[1]-outer_x[0])*(.47-i*.035)
        path = _bezier((center_x-half,y),(center_x,y-height*.012),(center_x+half,y+height*.003))
        path[:,1] += np.sin(np.linspace(0,2*np.pi,len(path))+i)*height*.0040
        paths['forehead_wrinkles'].append(path)
    # Lateral canthal lines radiate from outer corners33/263; they do not cross
    # the eye opening. The protected mask provides a second safety boundary.
    for corner,sign in ((33,-1),(263,1)):
        origin = p[corner]+[sign*width*.012,height*.002]
        for yoffset,length in ((-.045,.085),(-.018,.11),(.022,.10),(.056,.075)):
            end = origin+[sign*width*length,height*yoffset]
            paths['crow_feet'].append(_bezier(origin,origin+[sign*width*length*.4,height*yoffset*.2],end))
    # Short, staggered glabellar furrows stay between/just above the brows.
    # Long parallel brow-to-hairline strokes are intentionally not used.
    for sign,offset in ((-1,-.012),(1,.013)):
        x = p[168,0]+sign*width*.016
        start = (x+sign*width*.006,brow_y-height*(.038+offset))
        middle = (x-sign*width*.004,brow_y-height*.004)
        end = (x+sign*width*.003,brow_y+height*(.023+offset))
        paths['glabellar_lines'].append(_bezier(start,middle,end))
    # Alar98/327 to oral commissures61/291 define the nasolabial folds.
    for alar,corner,sign in ((98,61,-1),(327,291,1)):
        a = p[alar]+[sign*width*.012,height*.005]
        end = p[corner]+[sign*width*.045,height*.05]
        paths['nasolabial_folds'].append(_bezier(a,a+[sign*width*.055,height*.07],
            p[corner]+[sign*width*.075,-height*.065],end))
        start = p[corner]+[sign*width*.025,height*.018]
        paths['marionette_lines'].append(_bezier(start,start+[sign*width*.025,height*.085],
            start+[sign*width*.005,height*.095]))
    # Short perioral creases stop before the protected lip polygon.
    mouth_width = abs(p[291,0]-p[61,0])
    for t in (.14,.27,.40,.60,.73,.86):
        x = p[61,0]*(1-t)+p[291,0]*t
        paths['perioral_lines'].append(_bezier((x,p[0,1]-height*.0352),
            (x+(t-.5)*width*.01,p[0,1]-height*.035),(x,p[0,1]-height*.02)))
    for t in (.23,.4,.6,.77):
        x = p[61,0]*(1-t)+p[291,0]*t
        paths['perioral_lines'].append(_bezier((x,p[17,1]+height*.015),
            (x,p[17,1]+height*.04),(x+(t-.5)*mouth_width*.04,p[17,1]+height*.053)))
    for outer,lower,inner in ((33,145,133),(263,374,362)):
        a,b,c = p[outer],p[lower],p[inner]
        paths['under_eye_bags'].append(_bezier(a+[0,height*.035],b+[0,height*.085],c+[0,height*.035]))
        paths['tear_trough'].append(_bezier(c+[0,height*.025],
            (c+b)/2+[0,height*.055],b+[0,height*.065]))
    return paths


def _furrow_height(shape, paths, face_height: float, *, depth: float, width: float) -> np.ndarray:
    """A tapered depression plus paired tissue rolls, expressed in image pixels.

    Paths seed an actual height field. They are not painted onto RGB. Distance
    and nearest-seed weights produce continuously varying furrow width/depth;
    each path has a different low-frequency irregularity and tapered endpoints.
    """
    seeds = np.zeros(shape,np.uint8)
    for index,path in enumerate(paths):
        q = path.copy()
        tangent = np.gradient(q,axis=0)
        length = np.maximum(np.linalg.norm(tangent,axis=1),1e-6)
        normal = np.column_stack((-tangent[:,1],tangent[:,0]))/length[:,None]
        t = np.linspace(0,1,len(q))
        variation = np.sin(t*np.pi*3.3+index*1.9)*face_height*.0018
        q += normal*variation[:,None]*np.sin(np.pi*t)[:,None]
        q = np.rint(q).astype(np.int32)
        fade = np.sin(np.pi*t)**.7
        amplitude = fade*(.78+.22*np.sin(t*11+index)**2)
        for i in range(len(q)-1):
            cv2.line(seeds,tuple(q[i]),tuple(q[i+1]),int(255*amplitude[i]),1,cv2.LINE_AA)
    if not seeds.any():
        return np.zeros(shape,np.float32)
    binary = np.where(seeds>16,0,255).astype(np.uint8)
    distance,labels = cv2.distanceTransformWithLabels(binary,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    weights = np.zeros(int(labels.max())+1,np.float32)
    on_path = binary==0
    weights[labels[on_path]] = seeds[on_path]/255
    taper = weights[labels]
    size = max(face_height/300,1e-3)
    # A smooth local width modulation avoids uniform machine-drawn trenches.
    sigma = max(.45,width*size)*( .82+.18*taper )
    furrow = -np.exp(-.5*(distance/sigma)**2)
    roll = .33*np.exp(-.5*((distance-2.35*sigma)/(1.2*sigma))**2)
    return ((furrow+roll)*taper*depth*size).astype(np.float32)


def _relight_height(source: np.ndarray, height: np.ndarray, mask: np.ndarray,
                    face_height: float) -> tuple[np.ndarray,dict]:
    """Source-driven diffuse normal relief, ambient occlusion and tiny parallax.

    This is an explicit procedural diagnostic, not learned human skin anatomy.
    Broad source lighting/color is retained; only local relief response is added.
    """
    h,w = height.shape
    yy,xx = np.indices((h,w),dtype=np.float32)
    luma = source @ np.array([.2126,.7152,.0722],np.float32)
    eligible = mask>.45
    direction = np.array([-.32,-.48,.82],np.float32)
    if eligible.sum()>100:
        # Robust coarse luminance-plane fit; limit influence of highlights,
        # existing spots and model-independent source-camera color variation.
        coords = np.column_stack(((xx[eligible]-w/2)/max(w,1),
                                  (yy[eligible]-h/2)/max(h,1),np.ones(eligible.sum())))
        values = np.clip(luma[eligible],*np.percentile(luma[eligible],[8,92]))/255
        stride = max(1,len(values)//5000)
        fit = np.linalg.lstsq(coords[::stride],values[::stride],rcond=None)[0][:2]
        magnitude = float(np.linalg.norm(fit))
        if magnitude>1e-5:
            blend = min(.7,magnitude/.16)
            direction[:2] = (1-blend)*direction[:2]+blend*.55*fit/magnitude
            direction[2] = np.sqrt(max(.3,1-float(direction[:2]@direction[:2])))
    direction /= np.linalg.norm(direction)
    gy,gx = np.gradient(height)
    norm = np.sqrt(1+gx*gx+gy*gy)
    diffuse = (-gx*direction[0]-gy*direction[1]+direction[2])/norm
    response = (.64+.36*np.clip(diffuse,0,1))/(.64+.36*direction[2])
    size = max(face_height/300,1e-3)
    surrounding = cv2.GaussianBlur(height,(0,0),max(.8,face_height*.006))
    ambient_occlusion = np.clip((surrounding-height)/(1.2*size),0,1)*.075
    horizontal = max(float(np.linalg.norm(direction[:2])),1e-5)
    lx,ly = direction[:2]/horizontal
    slope = direction[2]/horizontal
    occlusion = np.zeros_like(height)
    for step in (.45,.9,1.8):
        reach = step*size
        neighbor = cv2.remap(height,xx+lx*reach,yy+ly*reach,cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        occlusion = np.maximum(occlusion,neighbor-height-reach*slope)
    self_shadow = np.clip(occlusion/max(size,1e-3),0,1)*.06
    response = np.clip(response-ambient_occlusion-self_shadow,.68,1.17)
    # Transport the source pore/color texture minutely along the local relief;
    # no new pixel content is sampled from another photo or a global cache.
    parallax = .22
    displaced = cv2.remap(source,xx+np.clip(gx*parallax,-.45,.45),
        yy+np.clip(gy*parallax,-.45,.45),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return displaced*response[...,None],{'light_direction':direction.tolist(),
        'height_range_px':[float(height.min()),float(height.max())],
        'relief_technique':'multiscale height normals with source-driven diffuse lighting, ambient occlusion and skin parallax'}


def _semantic_mask(value: np.ndarray, shape, label: str) -> np.ndarray:
    mask = np.asarray(value,dtype=np.float32)
    if mask.shape != shape or not np.isfinite(mask).all():
        raise ValueError(f'{label} must be a finite image-sized array')
    if mask.max(initial=0) > 1:
        mask = mask/255
    return np.clip(mask,0,1)


def regional_appearance_weights(image_shape,p,scores):
    """Soft region controls on a learned residual; never hand-drawn creases."""
    h,w=image_shape[:2];fw,fh=np.ptp(p,axis=0);y,x=np.ogrid[:h,:w]
    weight=np.full((h,w),max(scores.get('facial_wrinkles',0)*.8,
        scores.get('skin_texture',0)*.8),np.float32)
    def add(name,centers,rx,ry):
        nonlocal weight
        value=scores.get(name,0)
        if not value:return
        for center in centers:
            field=np.exp(-.5*(((x-center[0])/(fw*rx))**2+((y-center[1])/(fh*ry))**2))
            weight=np.maximum(weight,field*value)
    add('forehead_wrinkles',[(p[10]+p[9])/2],.40,.18)
    add('glabellar_lines',[p[9]],.14,.12)
    add('crow_feet',[p[33]+[-fw*.05,0],p[263]+[fw*.05,0]],.16,.10)
    for name in ('nasolabial_folds','deep_nasolabial_folds'):
        add(name,[(p[4]+p[61])/2,(p[4]+p[291])/2],.16,.18)
    add('marionette_lines',[p[61]+[0,fh*.09],p[291]+[0,fh*.09]],.13,.14)
    add('perioral_lines',[(p[0]+p[17])/2],.25,.15)
    for name in ('under_eye_bags','tear_trough'):
        add(name,[p[145]+[0,fh*.045],p[374]+[0,fh*.045]],.18,.09)
    add('cheek_volume_loss',[p[50],p[280]],.24,.19)
    add('skin_laxity',[p[136],p[365]],.24,.17)
    return np.clip(weight,0,1)


def _render_learned_skin(image: np.ndarray, p: np.ndarray, plan: dict,
                         proposal: np.ndarray, *, skin_mask=None, neck_mask=None):
    """Transfer aligned learned skin luminance residual; never draw crease paths.

    The neural generator is a channel-level appearance proposal, not an execution
    trace for individual knowledge-graph features. Source chroma is preserved and
    only source/proposal images with the SAME geometry may be passed here.
    """
    if proposal.shape != image.shape or proposal.dtype != np.uint8:
        raise ValueError('Learned proposal must be a matching RGB uint8 image')
    scores = {name:_score(plan,name) for name in (*SUPPORTED_FEATURES,'facial_wrinkles','deep_nasolabial_folds')}
    neck_score = _score(plan,'neck_lines')
    requested = [name for name,score in scores.items() if score > 0]
    if neck_score > 0:
        requested.append('neck_lines')
    strength = max(scores.values(),default=0)
    metadata = {'features_rendered':[], 'applied_features':[], 'requested_features':requested,
        'mean_abs_change':0.0, 'changed_pixels':0, 'unsupported_features':[],
        'mode':'learned_skin_appearance', 'source_chroma_preserved':True,
        'hand_authored_crease_paths_used':False,
        'attribution':'Channel-level learned appearance; individual graph features are not pixel-verified.'}
    if strength == 0 and neck_score == 0:
        return image.copy(),metadata
    skin = skin_region_mask(image.shape,p)
    if skin_mask is not None:
        skin *= _semantic_mask(skin_mask,image.shape[:2],'Skin mask')
    alpha = skin*regional_appearance_weights(image.shape,p,scores)
    if neck_score > 0:
        if neck_mask is None:
            metadata['unsupported_features'].append('neck_lines')
        else:
            neck = _semantic_mask(neck_mask,image.shape[:2],'Neck mask')
            y,x = np.ogrid[:image.shape[0],:image.shape[1]]
            width,height = np.ptp(p,axis=0)
            neck *= ((y >= p[152,1]-height*.015) & (y <= p[152,1]+height*.55) &
                     (np.abs(x-p[152,0]) <= width*.42))
            if neck.any():
                alpha = np.maximum(alpha,neck*neck_score)
            else:
                metadata['unsupported_features'].append('neck_lines')
    if not (alpha>0).any() or np.array_equal(image,proposal):
        return image.copy(),metadata
    yy,xx = np.nonzero(alpha>0)
    x0,y0,x1,y1 = int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1
    original = image[y0:y1,x0:x1]
    target = proposal[y0:y1,x0:x1]
    region_alpha = alpha[y0:y1,x0:x1]
    source_lab = cv2.cvtColor(original,cv2.COLOR_RGB2LAB)
    target_lab = cv2.cvtColor(target,cv2.COLOR_RGB2LAB)
    source_l = source_lab[...,0].astype(np.float32)
    target_l = target_lab[...,0].astype(np.float32)
    valid = region_alpha > float(region_alpha.max())*.35
    source_q = np.percentile(source_l[valid],[25,50,75])
    target_q = np.percentile(target_l[valid],[25,50,75])
    # Match the global skin tone/illumination, but allow at most a 5% contrast
    # adjustment so learned wrinkle and local volume cues are not normalized away.
    contrast = float(np.clip((source_q[2]-source_q[0]+6)/(target_q[2]-target_q[0]+6),.95,1.05))
    corrected = (target_l-float(target_q[1]))*contrast+float(source_q[1])
    residual = corrected-source_l
    # Retain broad volume shading and mid-scale furrows; attenuate only the
    # finest residual detail to reduce the papery texture of tiled inference.
    # No hand-authored crease paths run in this learned transfer.
    fine = residual-cv2.GaussianBlur(residual,(0,0),.9)
    lifestyle_gain = regional_gain(image.shape,p,plan['features'])[y0:y1,x0:x1]
    transferred = np.clip((residual-fine*.4)*lifestyle_gain,-55,55)
    metadata['lifestyle_gain_max'] = float(lifestyle_gain.max())
    metadata['lifestyle_localization'] = 'Authored regional gain on shared age proposal, not neural feature attribution.'
    edited_lab = source_lab.copy()
    edited_lab[...,0] = np.clip(np.rint(source_l+transferred),0,255).astype(np.uint8)
    candidate = cv2.cvtColor(edited_lab,cv2.COLOR_LAB2RGB).astype(np.float32)
    output = image.copy()
    output[y0:y1,x0:x1] = np.clip(np.rint(original.astype(np.float32)+
        (candidate-original)*region_alpha[...,None]),0,255).astype(np.uint8)
    difference = np.abs(output.astype(np.int16)-image.astype(np.int16))
    count = int(np.any(difference,axis=2).sum())
    metadata.update(mean_abs_change=float(difference.mean()),changed_pixels=count,
        mask_pixels=int((alpha>0).sum()),proposal_luminance_contrast_scale=contrast,
        proposal_luminance_median_shift=float(source_q[1]-target_q[1]))
    if count:
        metadata['features_rendered'] = ['learned_skin_appearance']
        metadata['applied_features'] = ['learned_skin_appearance']
    return output,metadata


def render_skin(image: np.ndarray, points: np.ndarray, plan: dict, *,
                skin_mask: np.ndarray | None = None,
                neck_mask: np.ndarray | None = None,
                learned_proposal: np.ndarray | None = None) -> tuple[np.ndarray,dict]:
    """Blend learned skin appearance, or run explicit procedural diagnostics.

    All scores are rendering deltas [0,1], not probabilities or clinical severity.
    Equal-age plans should have all scores zero and return an exact image copy.
    With learned_proposal supplied, ALL authored crease paths are bypassed and
    source chroma/protected regions are preserved. Without it, the diagnostic
    fallback renders a height/normal field, never simple RGB wrinkle strokes.
    The fallback's supersampled ROI delta is resized and masked again; no image
    is returned at a new size or written to disk.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Skin rendering expects an RGB uint8 image')
    p = _points(points)
    scores = {name:_score(plan,name) for name in SUPPORTED_FEATURES}
    if learned_proposal is not None:
        return _render_learned_skin(image,p,plan,learned_proposal,skin_mask=skin_mask,neck_mask=neck_mask)
    metrics = {'features_rendered':[], 'mean_abs_change':0.0, 'changed_pixels':0,
               'unsupported_features':['neck_lines'] if _score(plan,'neck_lines') > 0 and neck_mask is None else [],
               'mode':'authored_illustration'}
    neck_score = _score(plan,'neck_lines')
    if not any(scores.values()) and (neck_score == 0 or neck_mask is None):
        return image.copy(),metrics
    mask = skin_region_mask(image.shape,p)
    if skin_mask is not None:
        external = np.asarray(skin_mask,dtype=np.float32)
        if external.shape != image.shape[:2] or not np.isfinite(external).all():
            raise ValueError('Skin mask must be a finite image-sized array')
        if external.max(initial=0)>1:
            external = external/255
        mask *= np.clip(external,0,1)
    # A semantic neck mask is required: Face Mesh alone cannot distinguish a
    # neck from a scarf, collar or background. Restrict it to the region directly
    # beneath the chin before painting any age-related neck crease.
    neck = np.zeros(image.shape[:2],np.float32)
    if neck_mask is not None:
        external_neck = np.asarray(neck_mask,dtype=np.float32)
        if external_neck.shape != image.shape[:2] or not np.isfinite(external_neck).all():
            raise ValueError('Neck mask must be a finite image-sized array')
        if external_neck.max(initial=0)>1:
            external_neck = external_neck/255
        y,x = np.ogrid[:image.shape[0],:image.shape[1]]
        width,height = np.ptp(p,axis=0)
        eligible = ((y >= p[152,1]-height*.015) & (y <= p[152,1]+height*.55) &
                    (np.abs(x-p[152,0]) <= width*.42))
        neck = np.clip(external_neck,0,1)*eligible
        if neck_score:
            mask = np.maximum(mask,neck)
    if not (mask>0).any():
        return image.copy(),metrics
    h,w = image.shape[:2]
    yy,xx = np.nonzero(mask>0)
    x0,y0 = max(0,int(xx.min())-4),max(0,int(yy.min())-4)
    x1,y1 = min(w,int(xx.max())+5),min(h,int(yy.max())+5)
    if x1 <= x0 or y1 <= y0:
        return image.copy(),metrics
    source = image[y0:y1,x0:x1]
    face_height = max(float(np.ptp(p[:,1])),1)
    scale = min(4.0,max(1.0,360/face_height))
    roi_size = (max(2,round((x1-x0)*scale)),max(2,round((y1-y0)*scale)))
    enlarged = cv2.resize(source,roi_size,interpolation=cv2.INTER_CUBIC).astype(np.float32)
    local = (p-[x0,y0])*scale
    fw,fh = np.ptp(local,axis=0)
    shape = enlarged.shape[:2]
    height_field = np.zeros(shape,np.float32)
    local_mask = cv2.resize(mask[y0:y1,x0:x1],roi_size,interpolation=cv2.INTER_LINEAR)
    maps = _wrinkle_paths(local)
    # Depth/width values are authored diagnostic height-field pixels at a 300px
    # face. They are NOT anatomical units or wrinkle-depth measurements.
    relief = {'forehead_wrinkles':(.75,.8),'crow_feet':(.55,.65),
        'glabellar_lines':(.8,1.15),'nasolabial_folds':(1.35,2.3),
        'marionette_lines':(.85,1.9),'perioral_lines':(.32,.6),
        'under_eye_bags':(.6,1.4),'tear_trough':(.5,1.3)}
    for name,paths in maps.items():
        if scores[name] > 0:
            depth,width = relief[name]
            height_field += _furrow_height(shape,paths,fh,depth=depth,width=width)*scores[name]
    size = fh/300
    for ids,sign in (((50,101,205),-1),((280,330,425),1)):
        center = local[list(ids)].mean(axis=0)+[sign*fw*.015,fh*.025]
        height_field -= _gaussian(shape,center,(fw*.12,fh*.09))*6*size*scores['cheek_volume_loss']
    for eye in (145,374):
        center = local[eye]+[0,fh*.05]
        height_field += _gaussian(shape,center,(fw*.105,fh*.026))*1.6*size*scores['under_eye_bags']
    for jaw in (136,365):
        height_field += _gaussian(shape,local[jaw]+[0,-fh*.02],(fw*.09,fh*.065))*2.0*size*scores['skin_laxity']
    if neck_score > 0 and neck.any():
        neck_paths = []
        for offset in (.08,.19,.31,.43):
            center = local[152]+[0,fh*offset]
            neck_paths.append(_bezier(center+[-fw*.24,-fh*.015],center+[0,fh*.045],
                                     center+[fw*.24,-fh*.015]))
        local_neck = cv2.resize(neck[y0:y1,x0:x1],roi_size,interpolation=cv2.INTER_LINEAR)
        height_field += _furrow_height(shape,neck_paths,fh,depth=.9,width=1.9)*neck_score*local_neck
        metrics['features_rendered'].append('neck_lines')
    # Preserve and coarsen the image's own pore structure. Source detail guides
    # microrelief; no uniform random scratch pattern is laid over the skin.
    luma = enlarged @ np.array([.2126,.7152,.0722],np.float32)
    if scores['skin_texture']:
        pore = luma-cv2.GaussianBlur(luma,(0,0),max(.7,fh*.005))
        height_field += np.clip(pore,-12,12)*.012*size*scores['skin_texture']
    changed,physical = _relight_height(enlarged,height_field,local_mask,fh)
    metrics.update(physical)
    metrics['mode'] = 'procedural_height_relief_diagnostic'
    metrics['hand_authored_crease_paths_used'] = True
    if scores['skin_texture']:
        fine = enlarged-cv2.GaussianBlur(enlarged,(0,0),max(.6,fh*.003))
        medium = cv2.GaussianBlur(enlarged,(0,0),max(.6,fh*.003))-cv2.GaussianBlur(enlarged,(0,0),max(1.0,fh*.018))
        changed += np.clip(fine*.20+medium*.12,-8,8)*scores['skin_texture']
    # Pigment is a separate optical transmission cue, not a painted crease.
    rng = np.random.default_rng(47123)
    if scores['age_spots']:
        spots = np.zeros(shape,np.float32)
        for _ in range(38):
            center = (rng.uniform(local[:,0].min(),local[:,0].max()),
                      rng.uniform(local[:,1].min(),local[:,1].max()))
            radius = rng.uniform(.0035,.011)*fw
            spots += _gaussian(shape,center,(radius,radius*rng.uniform(.7,1.3)))*rng.uniform(.035,.13)
        concentration = np.minimum(spots,.22)*scores['age_spots']
        changed *= np.exp(-concentration[...,None]*np.array([.72,1.05,1.35],np.float32))
    delta = cv2.resize(changed-enlarged,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)
    delta *= mask[y0:y1,x0:x1,None]
    output = image.copy()
    output[y0:y1,x0:x1] = np.clip(np.rint(source.astype(np.float32)+delta),0,255).astype(np.uint8)
    difference = np.abs(output.astype(np.int16)-image.astype(np.int16))
    metrics.update(features_rendered=[name for name,value in scores.items() if value > 0]+metrics['features_rendered'],
                   mean_abs_change=float(difference.mean()),
                   changed_pixels=int(np.any(difference,axis=2).sum()),
                   mask_pixels=int((mask>0).sum()),face_supersample=float(scale))
    return output,metrics
