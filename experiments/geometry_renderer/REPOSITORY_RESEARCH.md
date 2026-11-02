# AVATAR local aging renderer repository research

## Tested appearance providers

- Existing MediaPipe468 + OpenCVDelaunay warp remains the primary geometry path. It cannot synthesize new wrinkles/hair by itself; smooth highfrequency detail and lighting need separate rendering.
- [Fast-AgingGAN](https://github.com/HasnainRaz/Fast-AgingGAN): officialMIT model11.47MB, pinnedGitHubsource and weights verified, exportedONNX and numericallychecked Torch/OpenCV to max3.4e-6. CPU warm~0.35s512 but actualNASA output weakly aged, so **rejected asdefault on visualquality**, independent ofspeed. Repeated2pass worsensblotchyredness; do not chain it forstronger aging.
- [FRAN-style independent reimplementation](https://github.com/timroelofs123/face_reaging): MIT, **not Disney proprietaryofficialmodel**. PublicONNX conversion release at https://github.com/georgegach/facestudio/releases/tag/models is separately documentedMITmodel; no GPLapplicationcode copied. Model124230737bytes SHA2565941608e21f5d218ec52862938898efb276b63048aa8586372ff046ef865e22b verified. ExistingOpenCVCPUexecutes source/targetage-conditioned RGB residual, no newruntimepackage. CloudCPU single512~1.4–2.3s,1024sliding9windows~14.1s. ActualNASA learnedskin is morephoto-like; slidingoutput stronger than single. Remains **optional pending actual notebook end-to-end<20s and quality gate**; cloudtimings do not prove user'slaptop budget. Modelresiduals can change identity/color/lighting, require semantic skin mask and explicitlimits; KGcontrols outside model, no causaltrace claim.

## Geometry/detail implementation references

# Mesh-first facial-aging implementation survey

Read-only official repository/README/license review, 2026-11-02. No models/datasets installed or downloaded for this survey, no participant images used, no timing claim below is a new AVATAR measurement. Reconstruction methods do not become aging models merely by producing a dense mesh or normal map.

| Repository | What is actually useful | Code / asset license and prerequisites | CPU / integration assessment |
|---|---|---|---|
| [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) | Fast dense 3DMM fitting (~38,365 vertices), depth, UV texture mapping and CPU rasterizer. A plausible lightweight upgrade to geometry/normal estimation, not wrinkles or age progression. | Code MIT. MobileNet checkpoint and BFM-derived `configs/bfm_noneck_v3.pkl` are required; verify underlying BFM asset rights separately rather than assuming code MIT licenses every model/data asset. Cython/C++ NMS/Sim3DR build; ONNX Runtime supported. | Official README reports 1.35 ms 3DMM regression on i5-8259U/4 threads, face detection ~15 ms and dense reconstruction ~1 ms. These are upstream measurements, not AVATAR tests. Strongest practical CPU geometry candidate, but fitting someone does not create an older version of them. |
| [FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) | Parametric head shape, expression, jaw/neck/eyeball articulation; useful stable 3D topology and UV coordinates for age-displacement/normal fields. | Wrapper code MIT. Downloading model requires signup/agreement and landmark embeddings; current README describes FLAME as CC BY 4.0, but exact downloaded version/model terms must be checked separately. Python3.7/PyTorch1.4 reference stack. | Linear blendshape evaluation is a reasonable CPU target, but image fitting/rendering is extra work. No trained age dimension or gray-hair/wrinkle-generation behavior is provided. Better medium-term infrastructure than a drop-in aging result. |
| [DECA](https://github.com/yfeng95/DECA) | Coarse FLAME geometry + detailed displacement, estimated lighting, animatable expression wrinkles. Direct architectural reference for real geometric wrinkle relief instead of drawing dark curves. | Code/model custom noncommercial scientific license, not MIT. License restricts modification/redistribution/third-party availability without permission. FLAME registered asset, DECA checkpoint (Google Drive), optional BFM→FLAME albedo rights. PyTorch/PyTorch3D or JIT C++/CUDA rasterizer. | No verified laptop CPU <20 s result. Default research stack, asset access and license prevent treating it as an automatically vendorable dependency. Can study coarse+detail+lighting representation; do not copy its restricted code/models into AVATAR. Primarily reconstructs observed face/expression, not future age. |
| [EMOCA](https://github.com/radekd91/emoca) | DECA-derived emotional expression capture with improved eye/lip alignment and detail displacement. | Same kind of custom noncommercial scientific code/model license, FLAME assets, checkpoint download, optional BFM texture registration. PyTorch Lightning/conda research stack. Official README now marks EMOCA deprecated in favor of Inferno FaceReconstruction. | No demonstrated <20 s laptop CPU contract; more dependencies and less direct benefit for aging. Do not adopt for this prototype merely to get wrinkles: it reconstructs expression and inherits DECA restrictions. |
| [FaceScape](https://github.com/zhuhao-nju/facescape) | Real base geometry + 4K displacement/texture representation; 16,940 registered models from 847 people×20 expressions with age metadata. Useful research basis for how coarse geometry and wrinkle detail should be separated. | Dataset explicitly noncommercial research with strict participant-privacy and publication permissions; some face textures have eye mosaics, and only listed identities may be publishable. Asset access/license required; large dataset. Toolkit license is not a blanket license to redistribute identifiable scans/textures. | Dataset, not a ready local aging model. Age metadata is cross-sectional, not a validated individual aging trajectory. Do not use these participants as AVATAR test fixtures or copy their identity-specific textures without permission. Representation can guide an independent renderer. |
| [Detailed3DFace](https://github.com/yanght321/Detailed3DFace) | FaceScape paper implementation predicts single-expression or rigged displacement maps; demonstrates learned wrinkles as geometry on base mesh. | Code MIT; FaceScape bilinear assets separately noncommercial research/access-controlled, checkpoint from Google Drive. Python3.6/PyTorch1.6, reference demo uses `--gpu_ids 0`. Model fitting detector differs from paper because original detector could not be released. | No verified CPU latency or turnkey age control. Serious future comparison for displacement quality, but cannot promise laptop <20 s or solve age progression just by loading the reconstruction network. |

## Source-specific points

- DECA explicitly advertises shape/detail/lighting reconstruction and expression wrinkle animation. Its `fetch_data.sh` requires FLAME credentials and contains `--no-check-certificate`; do **not** execute that downloader as-is. TLS verification must remain enabled.
- EMOCA's per-project README explicitly says optional BFM albedo cannot be redistributed; users must obtain it and convert it themselves.
- FaceScape `doc/doc_tu_model.md` distinguishes base mesh from displacement, supplies 4K detail maps, and describes participant-specific publishable lists. These maps are not generic public-domain aging decals.
- `BFM_to_FLAME` is a conversion utility, not an aging implementation. Its [README](https://github.com/TimoBolkart/BFM_to_FLAME) says converted texture-space users must still follow BFM's model agreement.

## Recommendation for this AVATAR task

1. Keep the current local MediaPipe/OpenCV warp as the primary controllable geometry path. A denser reconstruction is optional; it does not fix missing texture on its own.
2. Build separate coarse geometry, high-frequency skin detail/normal/displacement, source-consistent lighting, and segmented hair channels. The knowledge graph specifies which visible features/regions are requested; it cannot certify that a learned detail map implements a biological mechanism.
3. If DECA-style details are desired long term, adopt the *representation* with independently authored code and properly licensed age/detail assets; do not transplant restrictive checkpoints or another person's scanned skin into the demo.
4. For the present <20-second/laptop budget, a measured optional age-conditioned FRAN skin residual is more practical than bringing in DECA/EMOCA. It must remain optional unless final local end-to-end performance and visual quality pass on the user's hardware. Fast-AgingGAN's weak sample result is not a sufficient default just because it is fast.
5. Identity/pose/light/expression consistency and actual appearance must be visually checked with licensed fixtures. A normal map, changed-pixel count, or positive triangle area alone does not demonstrate elderly appearance.

## Primary source URLs read

- https://github.com/cleardusk/3DDFA_V2/blob/master/readme.md
- https://github.com/cleardusk/3DDFA_V2/blob/master/LICENSE
- https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/README.md
- https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/LICENSE
- https://github.com/yfeng95/DECA/blob/master/README.md
- https://github.com/yfeng95/DECA/blob/master/LICENSE
- https://github.com/yfeng95/DECA/blob/master/fetch_data.sh
- https://github.com/radekd91/emoca/blob/master/README.md
- https://github.com/radekd91/emoca/blob/master/LICENSE
- https://github.com/radekd91/emoca/blob/master/gdl_apps/EMOCA/README.md
- https://github.com/zhuhao-nju/facescape/blob/master/README.md
- https://github.com/zhuhao-nju/facescape/blob/master/doc/doc_tu_model.md
- https://github.com/yanght321/Detailed3DFace/blob/master/README.md
- https://github.com/yanght321/Detailed3DFace/blob/master/LICENSE
