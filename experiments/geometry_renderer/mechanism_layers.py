"""Explicit, local phenotype operators for the source graph's aging vocabulary.

These are authored visual proxies, not biochemical simulation. No disease lesions
are generated: nodular_elastosis is a documented coarse-surface proxy only.
"""
from __future__ import annotations
import cv2
import numpy as np
from .skin_layers import skin_region_mask

SURFACE_GROUPS={
 'under_eye_circles':('under_eye_circles',),
 'pigmentation':('sunspots','age_spots','hyperpigmentation','inconsistent_facial_pigmentation','uneven_skin_tone'),
 'surface_texture':('dry_skin','skin_texture','enlarged_pores','nodular_elastosis'),
 'dullness':('dull_skin_tone',),
 'vascular':('facial_redness','broken_capillaries','spider_veins'),
}

def score(plan,name):
    return float(plan['features'].get(name,{}).get('score',0))


def volume_displacement(points,face_count,plan):
    face=points[:face_count];fw,fh=np.ptp(face,axis=0);cx=face[:,0].mean()
    delta=np.zeros_like(points,dtype=np.float64)
    def field(ids,rx,ry):
        centers=face[list(ids)]
        z=((face[:,None,0]-centers[None,:,0])/(fw*rx))**2+((face[:,None,1]-centers[None,:,1])/(fh*ry))**2
        return np.exp(-z*2).max(axis=1)
    # Midface support loss narrows the cheeks, rather than translating every
    # feature downward together. Lower-face laxity moves contour outward/down.
    hollow=field((50,101,205,280,330,425),.22,.20)*score(plan,'cheek_volume_loss')
    laxity=max(score(plan,'skin_laxity'),score(plan,'lower_face_sagging'))
    jaw=field((136,172,150,365,397,379),.17,.16)*laxity
    side=np.tanh((face[:,0]-cx)/(fw*.12))
    delta[:face_count,0] += -side*fw*.042*hollow+side*fw*.026*jaw
    delta[:face_count,1] += fh*.032*jaw
    puff=field((50,205,280,425),.16,.14)*score(plan,'facial_puffiness')
    delta[:face_count,0] += side*fw*.018*puff
    # Apparent nasal length shares the nose operator; max avoids double counting.
    return delta


def render_surface(image,points,plan,skin_mask=None):
    mask=skin_region_mask(image.shape,points)
    if skin_mask is not None: mask*=np.clip(skin_mask,0,1)
    if not mask.any(): return image.copy(),{'operators':{},'mode':'explicit_surface_proxies'}
    ys,xs=np.nonzero(mask>0);x0,x1=int(xs.min()),int(xs.max())+1;y0,y1=int(ys.min()),int(ys.max())+1
    p=points[:468]-[x0,y0];fw,fh=np.ptp(p,axis=0);roi=image[y0:y1,x0:x1].astype(np.float32);out=roi.copy()
    a=mask[y0:y1,x0:x1];y,x=np.ogrid[:roi.shape[0],:roi.shape[1]]
    def gauss(center,rx,ry):
        return np.exp(-.5*(((x-center[0])/(fw*rx))**2+((y-center[1])/(fh*ry))**2))
    operators={}
    def apply(name,change,ids):
        nonlocal out
        before=out.copy();out=np.clip(out+change*a[...,None],0,255)
        operators[name]={'features':list(ids),'changed_pixels':int(np.any(np.rint(out)!=np.rint(before),axis=2).sum()),
          'interpretation':'authored phenotype proxy, not clinical severity'}
    strength=score(plan,'under_eye_circles')
    eye=sum(gauss(p[i]+[0,fh*.045],.105,.035) for i in (145,374))
    apply('under_eye_circles',-eye[...,None]*np.array([13,14,10])*strength,('under_eye_circles',))
    rng=np.random.default_rng(9817)
    patches=np.zeros(roi.shape[:2],np.float32)
    for _ in range(72):
        center=[rng.uniform(p[:,0].min(),p[:,0].max()),rng.uniform(p[:,1].min(),p[:,1].max())]
        radius=rng.uniform(.004,.019)
        patches+=gauss(center,radius,radius*rng.uniform(.6,1.1))*rng.uniform(.12,.55)
    pig=max(score(plan,n) for n in SURFACE_GROUPS['pigmentation'])
    apply('pigmentation',-np.minimum(patches,.65)[...,None]*np.array([15,20,24])*pig,SURFACE_GROUPS['pigmentation'])
    lum=roi@np.array([.2126,.7152,.0722]);detail=lum-cv2.GaussianBlur(lum,(0,0),max(.6,fw*.005))
    noise=rng.normal(0,1,lum.shape).astype(np.float32)
    grain=cv2.GaussianBlur(noise,(0,0),.55)-cv2.GaussianBlur(noise,(0,0),2)
    tex=max(score(plan,n) for n in SURFACE_GROUPS['surface_texture'])
    apply('surface_texture',np.repeat((detail*.28+grain*2.8)[...,None],3,axis=2)*tex,SURFACE_GROUPS['surface_texture'])
    dull=score(plan,'dull_skin_tone')
    apply('dullness',((lum[...,None]-roi)*.10-2)*dull,('dull_skin_tone',))
    vascular=max(score(plan,n) for n in SURFACE_GROUPS['vascular'])
    cheeks=sum(gauss(p[i],.15,.12) for i in (50,280,4))
    vessels=np.zeros(lum.shape,np.uint8)
    for anchor in (50,280):
        center=p[anchor]
        for _ in range(5):
            offset=rng.uniform(-.06,.06,2)*[fw,fh]
            start=center+offset;end=start+[rng.uniform(-.05,.05)*fw,rng.uniform(.02,.055)*fh]
            cv2.line(vessels,tuple(np.rint(start).astype(int)),tuple(np.rint(end).astype(int)),150,1,cv2.LINE_AA)
    vessel=cv2.GaussianBlur(vessels.astype(np.float32)/255,(0,0),.65)
    apply('vascular',(cheeks*.25+vessel)[...,None]*np.array([9,-3,-3])*vascular,SURFACE_GROUPS['vascular'])
    result=image.copy();result[y0:y1,x0:x1]=np.rint(out).clip(0,255).astype(np.uint8)
    return result,{'operators':operators,'mode':'explicit_surface_proxies',
      'nodular_elastosis_representation':'coarse_surface_proxy_not_nodules'}


def ear_warp(image,regions,plan):
    """Semantic-ear lower-tissue deformation with positive inverse Jacobian.

    Hair-covered/missing ears are explicitly unavailable, never inferred from
    Face Mesh. Source masks move with pixels; no external-ear landmark claim.
    """
    ears=np.asarray(regions.get('ears',np.zeros(image.shape[:2])),np.float32)
    strength=max(score(plan,'earlobe_elongation'),score(plan,'earlobe_ptosis'))
    meta={'available':bool((ears>.5).any()),'strength':strength,'changed_pixels':0,'safety_scale':1.}
    if not meta['available'] or strength==0: return image.copy(),dict(regions),meta
    h,w=ears.shape;y,x=np.indices((h,w),dtype=np.float32);flow=np.zeros((h,w),np.float32)
    count,labels,stats,_=cv2.connectedComponentsWithStats((ears>.5).astype(np.uint8))
    for label in range(1,count):
        ex,ey,ew,eh,area=stats[label]
        if area<10:continue
        local=(labels==label).astype(np.float32)
        lower=np.clip((y-ey)/max(eh,1)-.35,0,.65)/.65
        field=cv2.GaussianBlur(local*lower,(0,0),max(1,eh*.06))
        flow+=field*eh*.12*strength
    scale=1.
    for _ in range(20):
        if np.min(1-np.gradient(flow*scale,axis=0))>.3:break
        scale*=.5
    mapy=y-flow*scale;mapy[[0,-1],:]=y[[0,-1],:];mapy[:,[0,-1]]=y[:,[0,-1]]
    result=cv2.remap(image,x,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    changed={key:cv2.remap(value,x,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
      for key,value in regions.items() if isinstance(value,np.ndarray) and value.shape==(h,w)}
    meta.update(safety_scale=scale,changed_pixels=int(np.any(result!=image,axis=2).sum()))
    return result,{**regions,**changed},meta
