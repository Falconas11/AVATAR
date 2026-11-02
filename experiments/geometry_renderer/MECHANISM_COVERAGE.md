# v0.3 source-graph implementation coverage

Scope: existing v03 source graph, 9 mechanisms and25 renderable aging-related features. All source CSV bytes are preserved. These are qualitative visual controllers/proxies, not physiological simulations or validated age predictions.

## Mechanisms

- `auricular_tissue_laxity`: `earlobe_elongation`, `earlobe_ptosis`.
- `collagen_degradation`: `forehead_wrinkles`, `crow_feet`, `nasolabial_folds`, `facial_wrinkles`, `deep_nasolabial_folds`.
- `elastic_fiber_damage`: `skin_laxity`, `jowls`, `lower_face_sagging`.
- `mmp_activity`: `forehead_wrinkles`, `crow_feet`, `nasolabial_folds`, `skin_laxity`, `jowls`, `facial_wrinkles`, `deep_nasolabial_folds`, `lower_face_sagging`.
- `nasal_support_weakening`: `nasal_tip_ptosis`, `increased_nasal_length`.
- `oxidative_stress`: `forehead_wrinkles`, `crow_feet`, `nasolabial_folds`, `skin_laxity`, `jowls`, `facial_wrinkles`, `deep_nasolabial_folds`, `lower_face_sagging`.
- `pigmentation_dysregulation`: `age_spots`, `uneven_skin_tone`, `sunspots`.
- `skin_barrier_impairment`: `skin_texture`, `dry_skin`.
- `uv_dna_damage`: `age_spots`, `uneven_skin_tone`, `sunspots`.

## Feature operators

Neural entries mean regionally controlled learned residuals, not per-feature model attribution. Shared aliases are max aggregated. Availability is per-image; occluded ears are never fabricated.

| Source feature | Operator | Availability / limitation |
|---|---|---|
| `crow_feet` | regional_learned_skin_residual | implemented_visual_proxy |
| `deep_nasolabial_folds` | regional_learned_skin_residual | implemented_visual_proxy |
| `dry_skin` | explicit_surface_proxy | implemented_visual_proxy |
| `dull_skin_tone` | explicit_surface_proxy | implemented_visual_proxy |
| `facial_puffiness` | mesh_geometry | implemented_visual_proxy |
| `facial_wrinkles` | regional_learned_skin_residual | implemented_visual_proxy |
| `forehead_wrinkles` | regional_learned_skin_residual | implemented_visual_proxy |
| `jowls` | mesh_geometry | implemented_visual_proxy |
| `lower_face_sagging` | mesh_geometry | implemented_visual_proxy |
| `marionette_lines` | regional_learned_skin_residual | implemented_visual_proxy |
| `nasolabial_folds` | regional_learned_skin_residual | implemented_visual_proxy |
| `under_eye_circles` | explicit_surface_proxy | implemented_visual_proxy |
| `uneven_skin_tone` | explicit_surface_proxy | implemented_visual_proxy |
| `hyperpigmentation` | explicit_surface_proxy | implemented_visual_proxy |
| `inconsistent_facial_pigmentation` | explicit_surface_proxy | implemented_visual_proxy |
| `sunspots` | explicit_surface_proxy | implemented_visual_proxy |
| `nodular_elastosis` | explicit_surface_proxy | coarse_surface_proxy_not_nodules |
| `enlarged_pores` | explicit_surface_proxy | implemented_visual_proxy |
| `broken_capillaries` | explicit_surface_proxy | implemented_visual_proxy |
| `facial_redness` | explicit_surface_proxy | implemented_visual_proxy |
| `spider_veins` | explicit_surface_proxy | implemented_visual_proxy |
| `nasal_tip_ptosis` | mesh_geometry | implemented_visual_proxy |
| `increased_nasal_length` | mesh_geometry | implemented_visual_proxy |
| `earlobe_elongation` | semantic_ear_warp | implemented_visual_proxy |
| `earlobe_ptosis` | semantic_ear_warp | implemented_visual_proxy |

Each new mechanism/feature control has tests. Nodular elastosis is a coarse-surface proxy, not pathology synthesis. Graph references are copied without invented citations; all new numeric age curves remain explicitly provisional.
