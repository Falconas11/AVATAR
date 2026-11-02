# Face datasets for natural aging evaluation

Reviewed2026-11-02 (America/Chicago). Official project READMEs were read through
GitHub; [structured catalog](face_datasets.json) records their blob and content
hashes. No dataset photograph was downloaded, redistributed, used as a test
fixture or used to train/calibrate this renderer. The contour fix below is an
explicit rendering correction, not a learned rule extracted from these datasets.

| Dataset / inspected source | Relevant data | Role / limits |
|---|---|---|
| [FFHQ-Aging](https://github.com/royorel/FFHQ-Aging-Dataset) |70,000 FFHQ images plus10 age groups/confidence, pose, occlusion and semantic labels |Best candidate for age/pose-filtered appearance stress evaluation; age groups are not longitudinal within-person aging. Dataset CC BY-NC-SA4.0 plus original per-image terms. |
| [FFHQ](https://github.com/NVlabs/ffhq-dataset) |70,000 faces at1024×1024, varied hair/background/accessories |High-resolution hair and texture evaluation; not paired age trajectories. Dataset CC BY-NC-SA4.0; image permissions/attribution vary. Author excludes recognition-system development. |
| [UTKFace](https://github.com/aicip/UTKFace) |20k+ faces, ages0–116,68-point landmarks |Adult age/pose/landmark stress tests after rights review. Non-commercial research; source image rights remain with owners. Labels were DEX-estimated and human-checked, not verified birth records. |
| [All-Age-Faces](https://github.com/JingchunCheng/All-Age-Faces-Dataset) |13,322 faces, ages2–80, landmarks; README says mostly Asian |Potential landmark/proportion evaluation. Each image is a different individual, so comparing age groups does not identify how one face changes. README does not establish general adaptation/redistribution permission; clarify first. |
| [FairFace](https://github.com/joojs/fairface) |Age and attribute labels; README states CC BY4.0 |Future balanced evaluation strata, not demographic inference at runtime or a source of personal aging deformations. Verify provenance/intended use. |

AgeDB, MORPH and controlled older-adult FACES were also considered. Their provider
pages were blocked by the environment's network access policy during this review;
their current access/license terms were not verified. FGNET/CACD search results
included third-party mirrors or limited reference-code pages; those were not used
as a substitute for authoritative dataset rights. No photo mirrors were downloaded.

## Why the faces became square

The old explicit renderer narrowed the midface while adding uniform outward
lower-jaw displacement and substantial overlapping downwards shifts. Even on
an originally narrow face, this increased relative lower-face width. The
knowledge graph does not provide a quantitative rule that every aging face should
become square. Cross-sectional datasets also cannot establish that rule: age,
individual identity, pose, expression, lighting and source selection are mixed.

## Implemented v0.4.3 correction

- Remove generic outward jaw widening. Retain modest local soft-tissue descent;
  reduce aggressive cheek hollowing. Independent cheek/jowl controls replace
  shared maximum strength that previously let the nose slider activate volume.
- Work in a source-face coordinate frame, so a tilted face does not receive a
  rigid screen-down deformation.
- Keep source-specific cheek/jaw/chin/lower-face ratios within conservative
  relative bounds. If the requested field exceeds those bounds, scale the whole
  field continuously before the existing triangle guard. The naturalization
  profile reduces cheek/jowl field gains; the standalone diagnostic retains its
  existing visibility behavior.
- Report source/requested/guarded/rendered ratios and guard scale in the UI.

These bounds are **authored engineering safeguards**, not anthropometric aging
rates or a biometric identity score. Limits are1.5% for cheek width and two
jaw/cheek ratios,2.5% for chin/jaw taper,4% for lower-face/height ratio. We do not
force faces toward a population mean or infer demographic attributes from them.
True aging can change these ratios more; this conservative demo favors avoiding
unwanted shape drift until calibrated evidence is available.

## Evaluation plan for approved datasets

Prefer adult, frontal/near-frontal, neutral-expression, high-quality subsets;
keep pose and expression matched, and report source-to-output contour ratios,
warp Jacobians, mask failures and blinded human naturalness/identity assessments
separately from age-group labels. Use held-out subjects where identifiers exist;
without identifiers, do not claim a subject-disjoint or longitudinal evaluation.
Image licenses and attribution must be resolved per applicable source before use
or redistribution. No dataset's aggregate mean face should replace the source
face's proportions. A dataset license is not evidence of model performance.

Current verification remains the four generated fictional inputs at matched
controls, synthetic coordinate stress cases, rotation/translation tests and
existing Python/browser regressions. This is not a completed real-world study.
