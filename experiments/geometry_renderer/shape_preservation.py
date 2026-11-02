"""Source-relative shape protection; authored safeguards, not aging statistics.

These ratios describe landmark geometry only. They are not biometric identity
scores, demographic priors or anthropometric calibration from a dataset.
"""
import numpy as np


def face_frame(points):
    p=np.asarray(points,dtype=np.float64)[:468]
    vertical=p[152]-p[10]
    vertical=vertical/max(float(np.linalg.norm(vertical)),1e-8)
    lateral=np.array([vertical[1],-vertical[0]])
    return lateral,vertical


def shape_ratios(points,frame=None):
    p=np.asarray(points,dtype=np.float64)[:468]
    lateral,vertical=frame or face_frame(p)
    def width(a,b):return max(abs(float((p[b]-p[a])@lateral)),1e-8)
    cheek=width(234,454);jaw=width(136,365)
    height=max(abs(float((p[152]-p[10])@vertical)),1e-8)
    return {'cheek_width':cheek,'jaw_to_cheek':jaw/cheek,
        'jaw_corner_to_cheek':width(172,397)/cheek,
        'chin_to_jaw':width(149,378)/jaw,
        'lower_face_to_height':abs(float((p[152]-(p[61]+p[291])/2)@vertical))/height}


LIMITS={'cheek_width':.015,'jaw_to_cheek':.015,'jaw_corner_to_cheek':.015,
        'chin_to_jaw':.025,'lower_face_to_height':.04}


def preserve_source_shape(points,delta,face_count=468):
    """Reduce a proposed field until source contour ratios stay within bounds.

    Keeps the direction/local pattern of the requested soft-tissue changes.
    Triangle orientation/area is checked separately by the existing mesh guard.
    Limits are conservative demo choices, not measured rates of human aging.
    """
    p=np.asarray(points,dtype=np.float64);d=np.asarray(delta,dtype=np.float64).copy()
    if p.shape!=d.shape or face_count!=468 or not np.isfinite(d).all():
        raise ValueError('Shape guard needs a finite matching 468-point face field')
    d[face_count:]=0
    frame=face_frame(p);source=shape_ratios(p,frame)
    def acceptable(scale):
        candidate=shape_ratios(p+d*scale,frame)
        return all(abs(candidate[k]/max(source[k],1e-8)-1)<=limit for k,limit in LIMITS.items())
    scale=1.
    if not acceptable(scale):
        low,high=0.,1.
        for _ in range(32):
            mid=(low+high)/2
            if acceptable(mid):low=mid
            else:high=mid
        scale=low
    return d*scale,{'scale':float(scale),'source':source,
        'requested':shape_ratios(p+d,frame),'guarded':shape_ratios(p+d*scale,frame),
        'relative_limits':dict(LIMITS),'clinically_calibrated':False,
        'interpretation':'Source-relative contour safeguard, not biometric identity verification or a dataset-trained age rule.'}
