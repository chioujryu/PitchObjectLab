# SAM3 Text-Guided Video Single-Target Tracker

This subproject uses Ultralytics SAM3 SDK to:

1. Use a **text prompt** to detect candidate objects in a video.
2. Select one target and keep tracking the **same object** over time.
3. Export a result video with persistent bounding box (and optional mask overlay).
4. Export a CSV log for per-frame tracking quality analysis.

## Features

- Text-prompt driven detection + tracking using `SAM3VideoSemanticPredictor`.
- Single-target lock with stable `target_id`.
- Re-identification fallback when original id is lost:
  - IoU continuity
  - appearance histogram similarity (HSV)
  - area consistency
- Temporal stabilization:
  - EMA box smoothing
  - optional Kalman filtering
- Visual output enhancements:
  - target bbox
  - optional segmentation mask overlay
  - optional drawing of all candidates for debugging
- Structured output:
  - tracked video (`.mp4`)
  - frame-level log (`.csv`)

## Project Structure

```text
projects/sam3_text_video_tracker/
├── README.md
├── config.yaml
├── main.py
└── requirements.txt
```

## Prerequisites

- Python 3.10+
- GPU recommended
- `sam3.pt` weights with approved access from Meta/Hugging Face

## Install

From repo root:

```bash
pip install -U ultralytics opencv-python pyyaml numpy
```

or

```bash
pip install -r projects/sam3_text_video_tracker/requirements.txt
```

## Configuration

Edit `projects/sam3_text_video_tracker/config.yaml`:

- `model`: absolute path of `sam3.pt`
- `source`: input video path
- `text_prompt`: prompt phrase, e.g. `"person with red shirt"`
- `output_video`: output rendered video path
- `output_csv`: per-frame tracking log path

Then tune:

- Tracking quality: `conf`, `iou`, `min_score`
- Detection stride: `vid_stride` (run detection every N frames, `1` = every frame)
- ReID: `reid_max_lost`, `reid_min_iou`, `reid_hist_weight`, `reid_area_weight`
- Smoothing: `use_ema_smoothing`, `ema_alpha`, `use_kalman`
- Visualization: `draw_mask`, `mask_alpha`, `draw_all_candidates`
- Output frames: `output_full_video`
  - `true`: export all frames from source video
  - `false`: export detection frames only (writer FPS will be adjusted to `fps / vid_stride`)

## Demo Command

### A) Run with config file

```bash
python projects/sam3_text_video_tracker/main.py \
  --config projects/sam3_text_video_tracker/config.yaml
```

### B) Override fields from CLI

```bash
python projects/sam3_text_video_tracker/main.py \
  --config projects/sam3_text_video_tracker/config.yaml \
  --source /path/to/video.mp4 \
  --text "red car" \
  --output projects/sam3_text_video_tracker/outputs/red_car.mp4 \
  --csv projects/sam3_text_video_tracker/outputs/red_car.csv \
  --vid-stride 3 \
  --det-only-video
```

Frame output modes:

- `--output-full-video`: force exporting all source frames
- `--det-only-video`: export only frames where detection is executed

## Output

- Video: object is continuously boxed across frames.
- CSV columns:
  - `frame_idx`
  - `target_id`
  - `x1,y1,x2,y2`
  - `score`
  - `lost_frames`
  - `status` (`tracked` / `lost`)
  - `text_prompt`

### Convert output video to H.264

If your player/IDE cannot preview the default output codec, convert the rendered video to H.264:

```bash
ffmpeg -i projects/sam3_text_video_tracker/outputs/tracked.mp4 -c:v libx264 -pix_fmt yuv420p -movflags +faststart projects/sam3_text_video_tracker/outputs/tracked_h264.mp4
```

## Recommended Tuning Strategy

1. Start with short clip (~10-30s).
2. Increase `conf` if false positives are many.
3. Increase `reid_max_lost` if occlusion is long.
4. Increase `reid_hist_weight` for better appearance lock.
5. Enable both `use_ema_smoothing` and `use_kalman` for shaky videos.
6. Turn on `draw_all_candidates` for debugging wrong target switches.

## Notes

- SAM3 is heavy; lower `imgsz` (e.g. 512) for speed.
- Prompt quality matters: use concrete noun phrases.
- For very crowded scenes, consider running multiple prompts and custom fusion logic.
