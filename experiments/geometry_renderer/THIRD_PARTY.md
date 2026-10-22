# Dependencies and asset provenance

The renderer and displacement field were authored for this experiment. No
third-party aging displacement code, morphometric rates, or age presets were
copied. Landmark numbers refer to MediaPipe's 468-point Face Mesh topology.

- **MediaPipe 0.10.21**: Apache License 2.0. The official wheel contains the
  Face Mesh/face detection TFLite assets used offline by the Solutions API.
  Upstream source, notices, models and topology:
  https://github.com/google-ai-edge/mediapipe/tree/v0.10.21/mediapipe/modules/face_landmark
  and https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/LICENSE .
  Face Mesh documentation: https://chuoling.github.io/mediapipe/solutions/face_mesh.html .
  This older pinned release deliberately retains the bundled-model Solutions
  interface; do not silently upgrade to a release requiring external Task models.
- **OpenCV 4.11**: Apache License 2.0; used for inverse remapping, mesh drawing,
  and image interpolation. https://github.com/opencv/opencv/blob/4.11.0/LICENSE
- **SciPy 1.15.3** (Qhull Delaunay) and **NumPy 1.26.4**: BSD-3-Clause with
  their bundled third-party notices. https://scipy.org and https://numpy.org .
- **Pillow 11.2.1**: HPND (PIL Software License).
  https://github.com/python-pillow/Pillow/blob/11.2.1/LICENSE
- **Streamlit 1.45.1**: Apache License 2.0. https://github.com/streamlit/streamlit/blob/1.45.1/LICENSE
- **pytest 8.3.5**: MIT License. https://github.com/pytest-dev/pytest/blob/8.3.5/LICENSE
- **NASA portrait of Eileen Collins**: public domain according to scikit-image's
  source documentation. See [fixture provenance](tests/fixtures/README.md).
  The screenshots are explicitly generated from that fixture only. No participant
  photos are included, and no NASA or subject endorsement is implied.

Dependency wheels retain their upstream licenses/notices in the isolated
installation. The full Linux/Python 3.11 resolved versions are recorded in
`requirements-linux-py311.lock`; install-time network downloads are distinct from
runtime photo processing, which has no external service calls.
