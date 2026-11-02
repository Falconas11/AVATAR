"""Provisional scenario routing, separate from the immutable source knowledge graph.

Gains and spatial localization are authored visual controls, not epidemiological
coefficients, exposure doses, verified causation or neural feature attribution.
"""
from __future__ import annotations
import numpy as np

# An explicit allowlist avoids treating every reachable path as permission for an
# effect. In particular, generic diet cannot stand in for vitamin deficiencies.
ROUTES = {
    'uv_exposure': {'forehead_wrinkles':.25,'crow_feet':.25,'facial_wrinkles':.20,
        'skin_laxity':.15,'jowls':.15,'lower_face_sagging':.15,
        'age_spots':.35,'sunspots':.35,'uneven_skin_tone':.30,'dry_skin':.20,'skin_texture':.20},
    'smoking': {'facial_wrinkles':.30,'crow_feet':.15,'nasolabial_folds':.20,
        'deep_nasolabial_folds':.20,'skin_laxity':.15,'jowls':.15,'lower_face_sagging':.15,
        'dry_skin':.25,'skin_texture':.25,'uneven_skin_tone':.15},
    'alcohol': {'dry_skin':.15,'skin_texture':.15,'facial_puffiness':.25,
        'facial_redness':.30,'broken_capillaries':.25},
    'stress': {'under_eye_circles':.15,'dull_skin_tone':.10,'dry_skin':.10,
        'enlarged_pores':.10,'skin_texture':.10},
    'diet': {},
}
LABELS = {'uv_exposure':'UV exposure','smoking':'Smoking','alcohol':'Alcohol',
          'stress':'Stress (exploratory)','diet':'Diet (not rendered)'}
PRESETS = {'Baseline (no added modifiers)':{},'UV exposure':{'uv_exposure':1.},
    'Smoking':{'smoking':1.},'Alcohol':{'alcohol':1.},
    'Stress (exploratory)':{'stress':1.},'Combined exposure':{'uv_exposure':1.,'smoking':1.,'alcohol':1.}}


def feature_contributions(feature, paths, values):
    """Only allowlisted features with an applicable source-root path can change."""
    roots={p['nodes'][0] for p in paths}
    return {factor:float(values.get(factor,0))*mapping[feature]
        for factor,mapping in ROUTES.items()
        if feature in mapping and factor in roots and values.get(factor,0)>0}


def scenario_evidence(features, values):
    records={}
    for factor,label in LABELS.items():
        affected={name:{'gain':f['lifestyle_contributions'][factor],
            'region':f['region'],'paths':[p for p in f['graph_paths'] if p['nodes'][0]==factor]}
            for name,f in features.items() if factor in f['lifestyle_contributions']}
        records[factor]={'label':label,'value':float(values.get(factor,0)),
            'features':affected,'status':'not_rendered_requires_specific_nutrition_evidence' if factor=='diet'
                else 'exploratory_graph_referenced_visual_scenario',
            'calibrated':False,'interaction_model':False,
            'note':'Repository reference labels are unverified; missing/provisional edges are retained. Gains and localization are authored. No dose, duration, deficiency or individual causation is inferred.'}
    return records


def regional_gain(shape, points, features):
    """Spatial gain on an existing learned residual, not synthetic wrinkle paths.

    Source facial_wrinkles is localized around mouth/eyes for the smoking
    scenario as an explicitly authored choice, not a location inferred by FRAN.
    Shared aliases use max; different regions never compound past 1.35.
    """
    h,w=shape[:2];p=points[:468];fw,fh=np.ptp(p,axis=0);y,x=np.ogrid[:h,:w]
    gain=np.zeros((h,w),np.float32)
    def field(centers,rx,ry):
        out=np.zeros((h,w),np.float32)
        for c in centers:
            out=np.maximum(out,np.exp(-.5*(((x-c[0])/(fw*rx))**2+((y-c[1])/(fh*ry))**2)))
        return out
    zones={
        'forehead_wrinkles':field([(p[10]+p[9])/2],.36,.15),
        'crow_feet':field([p[33],p[263]],.14,.10),
        'nasolabial_folds':field([(p[4]+p[61])/2,(p[4]+p[291])/2],.12,.15),
    }
    zones['deep_nasolabial_folds']=zones['nasolabial_folds']
    mouth=field([(p[0]+p[17])/2],.23,.13)
    exposed=np.maximum(zones['forehead_wrinkles'],field([p[50],p[280]],.22,.18))
    for name,f in features.items():
        if f['score']<=0:continue
        contributions=f.get('lifestyle_contributions',{})
        # Geometry and explicit color/texture effects run in their own operators.
        if name not in (*zones,'facial_wrinkles'):continue
        if name=='facial_wrinkles':
            local=mouth*contributions.get('smoking',0)+exposed*contributions.get('uv_exposure',0)
        else:local=zones[name]*min(.35,sum(contributions.values()))
        gain=np.maximum(gain,local)
    return 1+np.clip(gain,0,.35)
