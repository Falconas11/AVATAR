# Synthetic adult portrait fixtures

These four inputs were generated with OpenAI's image-generation tool in this
conversation at the user's request. They depict fictional adults, not research
participants or portraits of named real people. They are generated inputs only;
**all older-looking outputs are produced by the repository's local renderer**.

Generation specification: four different young-adult head portraits, roughly age30
appearance, front-facing, neutral closed mouth, uniform soft frontal studio light,
plain gray background, scalp/chin visible, no glasses, sharp natural skin detail.
The2×2 chart was split at its exact midpoint into four627×627 inputs. No enhancement,
face retouching, source-specific parameter adjustment or generated aged reference
was used. The original chart remains in the execution environment's generated
images folder; `manifest.json` pins its digest and every crop's digest/bounds.

Controls30→60/75 are synthetic experiment settings, not true or estimated ages.
Four generated faces do not establish real-world age accuracy, identity retention,
or demographic fairness. Generation can itself introduce facial/skin biases.

Reproduce the fixture-only batch (after installing local models):

```sh
python -m experiments.geometry_renderer.validate_synthetic_batch --quality ultra
```

Output images are explicitly written only by that validation command. The
interactive app continues to keep uploaded photos in memory without autosaving.
