# Engineering Playbook: Single-Camera, Gimbal-Mounted Detection & Tracking of a Persistently Tiny Soccer Ball on a Non-Standard Pickup Field

## TL;DR
- **Use a two-stack detector strategy** — a heatmap-based, multi-frame ball detector (WASB / TrackNet family) running at native resolution with SAHI tiling and a P2 high-resolution head — to keep recall high for a ball that is <10–15 px in every frame; pair it with a YOLO/RF-DETR player+ball detector for the multi-object scene.
- **Select the "main" ball by tracking-then-filter**: detect ALL balls, run a camera-motion-compensated multi-object tracker (BoT-SORT or UCMCTrack), bound the active area with grass segmentation (SAM2 / DeepLab / HSV) plus the player convex hull, then score each ball track by proximity to the densest player cluster (DBSCAN) and "being-played" cues, with hysteresis to prevent flicker.
- **On 4× RTX 5090** (each 32 GB GDDR7, 1,792 GB/s ≈ 1.79 TB/s bandwidth, 21,760 CUDA cores, 680 5th-gen Tensor Cores, 3,352 FP4 AI TOPS, 575 W, launched Jan 2025 at $1,999, **no NVLink — multi-GPU is PCIe-only**): the realtime pipeline uses 1 GPU for detection (TensorRT FP8/FP16), 1 for the heatmap ball net, 1 for segmentation/tracking, 1 as headroom; the offline pipeline parallelizes by video shards and adds RAFT optical flow, SAMURAI single-object recovery, and super-resolution (which lifted ball+player mAP by ~12% on degraded SoccerNet frames).

## Key Findings
1. **Standard bbox detectors fail on persistently tiny balls.** COCO defines "small" as area <32×32 px and detectors typically lose ~30 mAP points on small vs large objects; the tiny-object literature defines "tiny" as <16×16 px. For a few-pixel ball, heatmap detectors that consume multiple consecutive frames (WASB, TrackNet) are the proven approach — WASB evaluates soccer at a 4-pixel tolerance.
2. **Heatmap ball detectors lead the soccer-ball benchmark.** WASB (Tarashima, Abdul Haq, Wang & Tagawa, NTT Communications, BMVC 2023; HRNet backbone, 288×512 input, 3 consecutive frames, ~1.5M params) reaches **F1 88.2 / Accuracy 97.9 / AP 86.2** on the soccer dataset (Table 2, τ=4 px), beating TrackNetV2 (F1 86.6 / AP 77.2) and DeepBall (F1 44.5). But WASB by design outputs at most one ball per frame — a limitation for the multi-ball requirement.
3. **Camera motion compensation is mature and necessary.** BoT-SORT's GMC (ORB / sparse optical flow + RANSAC affine), UCMCTrack (uniform ground-plane compensation, >1000 FPS on a single CPU, motion-only), Norfair's optical-flow camera-motion estimator, and ECC/RAFT for offline all directly address the panning gimbal.
4. **Standard field registration will likely fail on a non-standard field.** SoccerNet-calibration methods detect line/keypoints and fit a homography to a regulation pitch template; with no regulation lines this breaks. The robust markings-free substitute is grass segmentation (HSV / green-chromaticity; a fine-tuned DeepLab reaching IoU 0.98; or SAM2/Mask2Former) intersected with the player convex hull.
5. **Main-ball selection is a solvable heuristic.** Player clustering (DBSCAN / convex hull / centroid) defines "where the action is"; ball-to-player proximity plus possession cues selects the game ball; temporal hysteresis / Hungarian assignment keeps the choice stable.
6. **4× RTX 5090 is ample** for both realtime and offline; the workload is detection-bound and embarrassingly parallel offline, and Blackwell FP8/FP4 tensor cores make heavy models (and super-resolution) cheap.

## Details

### (A) Persistent tiny-ball detection
The core problem: the ball is <10–15 px in EVERY frame. Three complementary techniques must be combined.

**A1. Heatmap, multi-frame ball detectors (primary).** The TrackNet family and WASB are purpose-built for "high-speed and tiny objects." They take several stacked consecutive frames and regress a Gaussian heatmap of ball location, learning motion/trajectory cues rather than per-frame appearance.
- **WASB** (Tarashima et al., BMVC 2023, arXiv 2311.05237, repo `nttcom/WASB-SBDT`): HRNet backbone with strides removed from the stem to preserve resolution, ~1.5M params, input 288×512, N=3 frames in / 3 heatmaps out. On the soccer benchmark it scores F1 88.2 / Acc 97.9 / AP 86.2 and beats six prior SOTA methods by 7.8–16.8% AP. Connected components above threshold δ=0.5 become ball candidates; an internal tracker enforces temporal consistency. **Critical limitation: WASB predicts at most one ball per frame and explicitly "cannot be applied to sports in which multiple balls are used simultaneously."** This must be worked around (see B). Note: WASB's "Soccer" benchmark is the D'Orazio/Spagnolo soccer dataset, NOT SoccerNet — a useful clarification when comparing numbers.
- **TrackNetV2** (288×512, 3-in/3-out MIMO, ~11.3M params): reported test accuracy/precision/recall ≈ 85.2%/97.2%/85.4%; ~31.8 FPS.
- **TrackNetV3** (`qaz812345/TrackNetV3`): adds background subtraction + trajectory rectification (interpolates occluded/missed positions); reports ~97.5% accuracy / 98.6% F1 on its shuttlecock test split.
- **TrackNetV4** (arXiv 2409.14543, ICASSP 2025): plug-and-play motion-attention maps via frame differencing on top of V2/V3; relative gains of +1.3/+1.2/+0.3/+0.7% in acc/prec/recall/F1. Absolute numbers are evaluation-protocol-sensitive (an independent reproduction found large discrepancies), so treat the relative improvement as the reliable claim.
- **BlurBall** (arXiv 2509.18387): extends WASB with motion-blur-aware heatmaps and velocity estimation — relevant because a fast tiny ball is usually blurred.

**A2. Tiling / SAHI at native resolution (recall booster).** Resizing a 1080p/4K frame to 640 destroys a few-pixel ball. SAHI (Slicing Aided Hyper Inference, arXiv 2202.06934, repo `obss/sahi`) slices the full-resolution frame into overlapping tiles (e.g., 640×640 with 0.2 overlap), runs the detector per tile, and merges with NMS/NMM. Reported gains of +6.8 AP (sliced inference) up to +12.7 AP (with slicing-aided fine-tuning) on aerial small-object benchmarks. SAHI is integrated with Ultralytics (YOLO11/YOLO26), MMDetection, and Roboflow `supervision` (`sv.InferenceSlicer`). Cost: many tiles per frame → use the 5090s' throughput; slice only the field ROI once the field is known (see D) to cut tile count.

**A3. High-resolution P2/P3 heads (architectural).** Standard YOLO heads are P3 (stride 8) / P4 / P5; the P3 head can in principle localize 8×8 px objects but downsampling erodes recall. Adding a **P2 head (stride 4, e.g., 160×160 feature map) extends coverage to ~4×4 px objects** and is the single most effective architectural change for sub-10-px targets (YOLO11-4K, SOD-YOLO, PPM-YOLOv11, GAME-YOLO all confirm this). Often the P5 head is dropped to save compute since there are no large objects of interest. This requires retraining (see H).

**A4. Super-resolution (offline, optional realtime).** Seweryn, Cheć, Łukasik & Wróblewska, "Improving Object Detection Quality in Football Through Super-Resolution Techniques" (arXiv 2402.00163), applied RLFN super-resolution before Faster R-CNN on degraded SoccerNet frames: **a 12% increase in mAP@0.50:0.95 for 320×240 frames upscaled fourfold**, and (in the ScienceDirect follow-up, S1877750326000669) **>21% mAP@0.50:0.95 when enlarging sixfold from 320×240 to 1920×1080**, with the largest benefit on "smaller objects such as balls and distant players." Use SR in the offline pipeline and, if latency allows, on the field-ROI crop in realtime (Blackwell FP4/FP8 makes SR cheap).

**A5. Temporal aggregation.** Beyond multi-frame input, exploit trajectory continuity: a Kalman filter on the ball track, TrackNetV3-style trajectory rectification/in-painting across occlusions, and confirming a candidate only if it is consistent across ≥2–3 frames. This suppresses the false positives that plague round-object detectors.

### (B) Multi-ball multi-object tracking
There are multiple identical-looking tiny balls (game ball + peripheral/adjacent-field balls). WASB/TrackNet alone cannot output multiple balls, so the architecture is:

**B1. Detect all balls.** Use a YOLO/RF-DETR detector with a `ball` class (plus the heatmap net's top-K candidates rather than its top-1) to enumerate every ball. Keep confidence threshold low to preserve recall, then rely on tracking + filtering to reject junk.

**B2. Multi-object tracking with identical, tiny objects.** Data association is the hard part: appearance ReID is useless (all balls look the same) and tiny boxes have near-zero IoU between frames under camera motion. Therefore favor **motion + position association, not appearance**:
- **BoT-SORT** (arXiv 2206.14651, repo `NirAharon/BoT-SORT`): Kalman filter + GMC; strong default; integrated in Ultralytics and BoxMOT.
- **OC-SORT / Deep OC-SORT**: observation-centric, robust to non-linear motion and brief occlusion; BoxMOT (`mikel-brostrom/boxmot`, `yolo_tracking`) added a **centroid-based cost function specifically "suitable for small and/or high speed objects and low FPS videos."**
- **UCMCTrack** (Yi et al., AAAI 2024, arXiv 2312.08952, repo `corfyi/UCMCTrack`): pure-motion, ground-plane Kalman filter with Mapped Mahalanobis Distance instead of IoU — explicitly designed so that "IoU fails as there is no intersection between bounding boxes," which is exactly the tiny-fast-ball case. Relying solely on motion cues it "ranks first on MOT17 without using any appearance cues" with "an exceptional speed of over 1000 FPS on a single CPU."
- **ByteTrack**: keeps low-confidence detections in a second association stage — valuable because tiny balls are often low-confidence.
- **Norfair**: lightweight, point/centroid-based, built-in camera-motion estimation and a SAHI demo; easy to add custom distance functions.

For identical tiny objects, IoU-based association should be replaced with **centroid/Mahalanobis distance after camera-motion compensation**. Expect ID switches when balls cross or occlude; this is acceptable because the downstream main-ball filter re-selects each frame with temporal smoothing.

**B3. Single-object tracker fallback for the main ball.** Once the main ball is identified, a single-object tracker can lock onto it through clutter: **SAMURAI** (arXiv 2411.11922) adapts SAM2 with motion-aware memory for zero-shot visual tracking of "fast-moving or self-occluding objects," real-time, no retraining; OSTrack/MixFormer are alternatives. Run SAMURAI/SAM2 on the field-ROI crop to recover the ball when the detector drops it.

### (C) Main-ball selection via player clustering
Goal: from N ball tracks, pick the single game ball = "closest to the cluster of players / currently being played."

**C1. Detect players, define "where the action is."** Detect all players (same YOLO/RF-DETR pass). Compute, per frame:
- **Densest cluster via DBSCAN** on player ground positions (DBSCAN is the standard in soccer spatial analysis; used to find dominant player groups and movement sequences). ε tuned per scene; the largest/densest cluster = the active play group.
- **Player centroid** and **convex hull** (the standard tactical "team block" / surface-area construct in soccer analytics, where ball-possession phases show larger convex-hull area). The action centroid is the centroid of the densest cluster, not all players (peripheral people shouldn't drag it).
- Optionally a kernel-density / mean-shift estimate of player density as a continuous "action heatmap."

**C2. Score each ball track.** Combine cues into a per-track score:
- **Proximity**: distance from ball to nearest player's feet and to the action centroid (smaller = better). The classic single-ball heuristic — "selects the ball closest to the [player] centroid, assuming there's only one ball on the field and its movement is physically realistic" — generalizes here.
- **Possession / being-played cues**: ball within a player-feet radius (ring-based possession as in sports-tracking patents and the Tryolabs possession method, where the player whose feet are closest within a threshold "has" the ball); ball motion correlated with a nearby player's action; recent possession continuity.
- **Inside-field bonus / outside-field veto** from (D).
- **Track quality**: track age/confidence and physically plausible velocity.

**C3. Stability over time (anti-flicker).** Do NOT pick argmax each frame independently. Apply:
- **Hysteresis**: only switch the "main ball" if a competitor's score exceeds the incumbent's by a margin for K consecutive frames.
- **Temporal smoothing** of scores (EMA) and **Hungarian assignment** of the "main-ball" label across time so it tracks one identity.
- Hold the last main ball through brief detection gaps (track buffer), bridging with the SOT (B3).

### (D) Markings-free field-boundary detection
**D1. Why standard registration fails.** SoccerNet camera-calibration / sports-field-registration methods (PnLCalib, the SoccerNet 2023 calibration winner, etc.) detect line intersections and field-marking keypoints, then fit a homography to a known regulation pitch model. These papers note intersection points "are marked with clear geometric markings," and homography annotations "often fail to account for lens distortions." On a casual field with no regulation lines and non-standard geometry, there is no template to register to — these methods cannot produce a reliable boundary.

**D2. Grass / playing-surface segmentation (markings-free).** Segment the green playing surface directly:
- **Color-based (cheap, fast)**: HSV/hue thresholding or green-chromaticity analysis; "A fully automatic method for segmentation of soccer playing fields" (Nature Sci. Reports 2023) combines normalized-green + RGB chromatic distortion with region-level post-processing to discard non-field green. RoboCup systems take the convex hull of green regions below the horizon as the field.
- **Learned segmentation (robust)**: a CNN grass segmenter — Homayounfar, Fidler et al., "Soccer Field Localization from a Single Image" (arXiv 1604.02715) fine-tuned DeepLab to **IoU 0.98** for grass vs non-grass. For zero-/few-shot, **SAM2** (promptable, video memory) or **Mask2Former/DeepLabv3+** segment the field surface and propagate the mask across frames. SAM2's memory bank makes it well-suited to maintaining the field mask while the camera pans.

**D3. Player convex hull as a markings-free field proxy.** The convex hull of all players (+ margin) bounds the active play area without any field appearance assumption — a robust complement when grass color is ambiguous (worn pitch, shadows, artificial turf). Combine: **active region = (grass mask) ∩ (dilated player convex hull)**. Balls outside this region (periphery, adjacent fields) are vetoed in C2.

**D4. Tracking the boundary under panning.** Since the gimbal pans continuously, the field mask must update. Two options: (i) **re-segment every N frames** (SAM2 propagation or fast HSV each frame); (ii) **warp the previous mask** using the per-frame homography/affine from the CMC module (E), re-segmenting only on drift. The cheap realtime choice is per-frame HSV grass + player-hull intersection; the accurate offline choice is SAM2 propagation.

### (E) Camera motion compensation (panning gimbal)
The background moves continuously, so static-camera assumptions break.
- **In-tracker GMC/CMC**: BoT-SORT estimates rigid camera motion by image registration between adjacent frames — keypoint extraction + sparse (Lucas-Kanade) optical flow with translation-based outlier rejection, solving an affine matrix via RANSAC, then correcting Kalman predictions. Methods: `files | orb | ecc` (and `sparseOptFlow`, the Ultralytics default "good balance of accuracy and efficiency for moving cameras"). Deep OC-SORT and StrongSORT include CMC modules; UCMCTrack folds camera motion into a uniform ground-plane model.
- **Standalone**: Norfair's `MotionEstimator` (optical flow on strong corners; "works for camera pans and tilts," keeps a fixed reference frame, updates it on drift). ORB/ECC image registration for affine/homography.
- **Offline / high accuracy**: **RAFT** dense optical flow for precise frame-to-frame motion; ECC for sub-pixel alignment. Use the estimated homography to (i) compensate tracker predictions, (ii) warp the field mask (D4), and (iii) optionally stabilize a fixed "world" coordinate frame for trajectory analysis.
- **Practical note**: because the ball and players are dynamic, estimate background motion from keypoints OUTSIDE detection boxes (sparse-registration outlier rejection does this); a panning-only motion is well-approximated by affine, so ORB+RANSAC affine is usually sufficient and cheap.

### (F) Realtime / online pipeline
Per-frame flow (target 25–30 FPS):
1. **Ingest** frame (1080p or 4K) → decode on GPU (NVDEC).
2. **CMC**: estimate affine/homography vs previous frame (ORB+RANSAC or sparseOptFlow).
3. **Field mask**: HSV grass segmentation (fast) warped/refreshed; intersect with player convex hull → active region. Restrict subsequent work to this ROI.
4. **Detection (GPU 0)**: YOLO26/YOLO11 with P2 head (TensorRT FP8/FP16) on the ROI, optionally SAHI-tiled, producing players + all ball candidates.
5. **Heatmap ball net (GPU 1)**: WASB/TrackNet (3-frame buffer) on the ROI for high-recall tiny-ball candidates; merge top-K with detector balls.
6. **Tracking (GPU 2/CPU)**: BoT-SORT (with GMC) or UCMCTrack on balls AND players; centroid/Mahalanobis association.
7. **Player clustering**: DBSCAN + densest-cluster centroid + convex hull.
8. **Main-ball selection**: score tracks (C2), apply hysteresis/EMA (C3); optionally hand the winner to SAMURAI SOT (GPU 2/3) for lock-on.
9. **Output**: main-ball coordinate + trajectory; (optionally feed back to drive the gimbal).

**GPU allocation (4× 5090)**: GPU0 detector, GPU1 heatmap ball net, GPU2 segmentation+tracking+SOT, GPU3 headroom/SR/redundancy or a second camera. The 5090's 32 GB and 1.79 TB/s bandwidth comfortably hold all models; use TensorRT (FP8 native on Blackwell; FP4 where stable) — NVIDIA reports TensorRT-for-RTX FP8/FP4 gives large speedups over FP16 on the 5090. Because the 5090 has **no NVLink**, treat the four cards as independent workers (one model per card) rather than relying on fast peer-to-peer model sharding.

### (G) Offline / post-processing pipeline
Accuracy-first, latency irrelevant; shard the video across the 4 GPUs (data-parallel) or run a model-parallel chain.
1. **Super-resolution** the field ROI (RLFN/diffusion, FP4) → +12–21% small-object mAP.
2. **Detection** with the largest models (RF-DETR-L/XL, YOLO26x, DEIM-D-FINE-X) + SAHI on full-res, lower thresholds.
3. **Heatmap ball net** (WASB, oversampling variant for best AP) over the whole clip.
4. **CMC with RAFT** dense flow + ECC for precise stabilization; build a global field coordinate frame.
5. **Offline MOT**: BoT-SORT/Deep OC-SORT (or BoostTrack), then **Gaussian-smoothed interpolation (GSI)** to fill gaps; bidirectional (forward+backward) tracking.
6. **Trajectory rectification** (TrackNetV3-style) and Kalman smoothing to in-paint occluded ball positions.
7. **SAMURAI/SAM2** for long-term single-ball recovery once the main ball is known; SAM2-Long memory tree for long clips.
8. **Main-ball selection** with full past+future context (non-causal hysteresis = cleaner than realtime).
9. **Evaluation** with TrackEval (HOTA/MOTA/IDF1) and SoccerNet dev-kit metrics.

### (H) Data & training
- **Datasets**: SoccerNet-Tracking (Cioppa, Giancola et al., CVPRW 2022, arXiv 2204.06918 — 201 sequences of 30s, 225,375 frames, 3,645,661 bounding boxes and 5,009 tracklets from 12 complete 2019 Swiss Super League games at 1080p/25fps, with players+referees+ball IDs), SoccerNet Ball Action Spotting (12 classes), SoccerNet-GameState; WASB-SBDT's soccer set (the D'Orazio/Spagnolo dataset, NOT SoccerNet). Roboflow has soccer ball/player datasets.
- **Domain gap**: all of the above are broadcast / high-mounted / panoramic; your camera is a low-mounted amateur gimbal. Expect a large domain gap — **you must collect and annotate your own footage** from the actual rig. Prioritize a few hours of representative pickup-game video.
- **Augmentation for tiny objects**: copy-paste of small balls (Kisantal et al., "Augmentation for small object detection" — oversampling + copy-paste lifts small-object AP), Mosaic (Ultralytics: "highly effective for improving small object detection"), MixUp/CutMix, and **synthetic camera-pan augmentation** (apply random affine/homography sequences to simulate the gimbal). Generative/diffusion copy-paste for rare configurations.
- **Detector training**: fine-tune YOLO26/YOLO11 with an added P2 head (drop P5), train at high input resolution; train the heatmap net (WASB) on your annotated ball positions with HLSM hard-sample mining. On 4× 5090 use DistributedDataParallel; 32 GB allows large-input/large-batch training.
- **Annotation strategy**: dense-label the ball (point annotations suffice for heatmap nets), players as boxes; semi-/self-supervised propagation (SAM2 to pre-label, human to correct) to scale; partially-labeled frames + cross-validation as TrackNet did.

### (I) Frameworks / repos
- **Detection**: Ultralytics (YOLO11/YOLO26, P2-head configs, built-in BoT-SORT/ByteTrack); RF-DETR (`roboflow`, Apache-2.0, DINOv2 backbone); D-FINE / DEIM; MMDetection.
- **Small objects**: SAHI (`obss/sahi`); Roboflow `supervision` (`InferenceSlicer`).
- **Ball-specific**: `nttcom/WASB-SBDT`; TrackNetV2/V3 (`qaz812345/TrackNetV3`); TrackNetV4; BlurBall.
- **Tracking**: BoxMOT / `yolo_tracking` (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, StrongSORT, BoostTrack, HybridSORT, ImprAssoc); `corfyi/UCMCTrack`; Norfair; motcpp (C++); TrackEval for metrics.
- **Segmentation / SOT**: SAM2, SAMURAI (`yangchris11/samurai`), Mask2Former, DeepLabv3+.
- **Sports**: SoccerNet dev-kits (`SoccerNet/sn-tracking`, `sn-spotting`); Roboflow `sports`; Tryolabs soccer-possession reference.
- **CMC / flow**: OpenCV (ORB, ECC, Lucas-Kanade, VideoStab GMC); RAFT.
- **Deploy**: TensorRT / TensorRT-LLM, NVDEC, Triton Inference Server for multi-GPU serving.

### (J) Concrete recommendations & caveats
**Recommended build order:**
1. Collect + annotate real rig footage first (biggest risk is domain gap).
2. Stand up the realtime baseline: YOLO26 (+P2, TensorRT FP8) + BoT-SORT (GMC) + HSV-grass∩player-hull + proximity/hysteresis main-ball selection. Measure recall on the tiny ball.
3. Add the WASB heatmap net in parallel to lift tiny-ball recall; merge candidates.
4. Add SAMURAI SOT lock-on for the main ball.
5. Build the offline pipeline (SR + big models + RAFT + GSI + non-causal selection) for ground-truth-quality output and to auto-label more training data.

**Thresholds that change the plan:**
- If ball recall <~70% with detector+SAHI alone → the heatmap net becomes mandatory, not optional.
- If ID-switch rate between balls is high → switch from IoU association to UCMCTrack ground-plane Mahalanobis or a centroid cost.
- If grass segmentation is unreliable (turf/shadows) → lean on player convex hull as the primary field bound.
- If realtime FPS <25 → restrict all heavy work to the field ROI, drop SR, use a smaller detector + heatmap net only.

**Caveats / uncertainties:**
- **WASB single-ball limitation** is real; the multi-ball requirement forces the top-K/merge workaround — this is not how WASB was validated, so test it.
- TrackNetV4 absolute accuracy numbers are protocol-dependent; rely on relative gains.
- 2026-specific model claims: YOLO26 (released Sept 2025, NMS-free, with Small-Target-Aware Label assignment/STAL) and YOLOv13 figures come largely from vendor/leaderboard sources (Roboflow leaderboard, Ultralytics) and should be re-benchmarked on your data; some arXiv "YOLO26" analyses carry 2026-dated IDs that warrant independent verification.
- SoccerNet "ball action spotting" is temporal action localization, not pixel-level ball localization — don't over-read its mAP numbers as small-ball detection accuracy.
- All cited benchmark numbers are from broadcast / standard-resolution data at 25–30 FPS; your low-mounted, panning, non-standard scenario will differ — validate everything on your own footage.
- The RTX 5090 has no NVLink and a 575 W TDP per card; size the chassis power/cooling for 4 cards accordingly and plan multi-GPU as independent per-card workers.