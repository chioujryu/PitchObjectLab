# RF-DETR Inference 足球跟蹤模組實作計畫

這份文件用來把「video inference 足球跟蹤」拆成可逐步交給 Claude Code 或 Codex 執行的模組。每個模組都應該可以單獨實作、測試，再進入下一個模組。

## 共同規則

- 只改 inference 相關行為；訓練、測試、推理 entrypoint 必須維持獨立。
- 主要入口仍是 `inference_rf_detr_model.py`。
- 新增 shared tracking 邏輯時放在新 helper module，不要讓 standalone train/test entrypoint import inference entrypoint。
- 不新增第三方套件。
- 所有新增 config 必須放在 `config/` 的 inference 設定裡，並保留 `runtime`, `model`, `dataset`, `output`, `inference`, `evaluation` 分組。
- 任何輸出行為都必須沿用現有 output estimate、confirmation、config snapshot、UTF-8 JSONL 寫入規則。
- 影片中同一 class ID 的 bbox 顏色必須在同一次 run 內保持一致。

## 預設規格

- Tracking 預設關閉：`inference.tracking.enabled: false`。
- 目標類別預設為足球：`target_class_names: [football]`。
- 足球 bbox 中心點落在 track 初始圓內，即視為同一顆足球。
- 初始圓的圓心固定為該 track 第一個足球 detection 的 bbox center，不隨後續 detection 更新。
- Track 不因中間漏檢幾個 frame 自動失效；`max_missing_frames: all`。
- 軌跡只在 video inference 渲染；image inference 不畫軌跡。

## Module 1: Config Schema

目標：先讓 config 明確描述 tracking 行為，但不改推理邏輯。

實作要求：

- 在 `config/rf_detr_inference.yaml` 的 `inference:` 下新增：

```yaml
  tracking:
    # Enable football tracking for video inference. Options: true, false.
    enabled: false
    # Category IDs to track. Empty means use target_class_names. Options: list of category IDs.
    target_class_ids: []
    # Category names to track. Empty with no target_class_ids defaults to football. Options: names such as [football].
    target_class_names:
      - football
    # Radius around the first detected ball center used for same-ball matching. Options: positive pixels.
    radius_pixels: 80
    # Maximum missing detection frames before a track expires. Options: all/null to never expire, or a non-negative integer.
    max_missing_frames: all
    # Draw trajectory lines for tracked balls in rendered videos. Options: true, false.
    draw_trajectory: true
    # Draw the fixed initial matching circle for each active track. Options: true, false.
    draw_anchor_circle: false
    # Width of trajectory lines in rendered video frames. Options: positive integer pixels.
    trajectory_width: 2
    # Include track ID in rendered football labels. Options: true, false.
    label_track_id: true
```

- 同步更新任何專用 inference config，至少包含目前的 `config/rf_detr_inference_sahi320_football_recheck640_video_1984090406020124673.yaml`。
- 第一版不新增 CLI override；tracking 只由 YAML 控制。

驗收：

- `yaml.safe_load()` 可解析所有更新後的 inference config。
- `tracking.enabled=false` 時，現有 inference 行為不變。

## Module 2: Pure Tracking Helper

目標：新增純邏輯 helper，先不接入 video pipeline。

新增檔案：

- `rf_detr_video_tracking.py`

資料結構：

- `TrackingConfig`
  - `enabled: bool`
  - `target_class_ids: set[int]`
  - `radius_pixels: float`
  - `max_missing_frames: Optional[int]`
  - `draw_trajectory: bool`
  - `draw_anchor_circle: bool`
  - `trajectory_width: int`
  - `label_track_id: bool`
- `TrackedBall`
  - `track_id: int`
  - `anchor_x: float`
  - `anchor_y: float`
  - `radius_pixels: float`
  - `first_frame_index: int`
  - `last_seen_frame_index: int`
  - `points: list[tuple[int, float, float]]`
- `TrackingAssignment`
  - `track_id: Optional[int]`
  - `center_x: float`
  - `center_y: float`
  - `is_target: bool`

核心函式：

- `parse_tracking_config(config, categories) -> TrackingConfig`
  - 從 `inference.tracking` 讀設定。
  - `target_class_ids` 優先；空值時用 `target_class_names` 解析；兩者都空時 fallback 到 `football`。
  - validate `radius_pixels > 0`, `trajectory_width > 0`, `max_missing_frames >= 0` 或 `all/null`。
- `bbox_center(prediction) -> tuple[float, float]`
  - 從 COCO xywh bbox 回傳中心點。
- `FootballTracker.update(frame_index, predictions) -> list[dict]`
  - 不修改輸入物件；回傳加上 tracking 欄位的新 prediction list。
  - 只處理 target football 類別。
  - 同一 frame 的足球 detection 先依 score 由高到低排序。
  - 每個 detection 只能分配到一條未使用 track。
  - 若 detection center 落入多個初始圓，選距離初始圓心最近者；距離相同選較小 `track_id`。
  - 無匹配時建立新 track。
  - 非 target prediction 的 tracking 欄位為 `None`。

驗收：

- 單元測試覆蓋同球、離開初始圓、新建 track、多球最近匹配、非足球不參與。
- 不需要載入 RF-DETR 模型即可測試。

## Module 3: Video Pipeline Integration

目標：讓 one-pass 與 batched video inference 都使用相同 tracking 邏輯。

實作要求：

- 在 `inference_rf_detr_model.py` 建立 `TrackingConfig` 與 `FootballTracker`。
- `tracking.enabled=false` 時不要建立有效 tracker，避免影響現有速度與輸出。
- `predict_video_file_one_pass()`：
  - 只在 `should_detect=true` 的 detection frame 呼叫 `tracker.update(absolute_frame_index, frame_predictions)`。
  - `all_predictions` 寫入 tracker 更新後的 rows。
  - skipped frames 沿用 `last_predictions` 與 tracker 已有軌跡渲染。
- `predict_video_file_batched()`：
  - detection pass 只保存 raw predictions by `segment_frame_index`。
  - render pass 依 frame 順序 replay：遇到 detection frame 時呼叫 tracker update，再用更新後 predictions render。
  - replay 後寫入 `all_predictions`，確保 JSONL 順序與渲染順序一致。
- batched 與 one-pass 在相同 prediction sequence 下必須產生一致的 `track_id`。

驗收：

- 現有 video batch size 行為保留。
- `render_skipped_frames=true` 時，skipped frames 可以看到最新 bbox 與累積軌跡。
- `render_skipped_frames=false` 時，只有 detection frames 被畫上 bbox/軌跡。

## Module 4: Rendering And Output Records

目標：在影片上畫足球軌跡，並把 tracking 欄位寫入 prediction records。

實作要求：

- 擴充 `draw_predictions()`，增加 optional tracking overlay 參數。
- bbox 顏色仍使用 `class_color(category_id)`。
- 軌跡線預設使用 football class color。
- 若 `draw_anchor_circle=true`，用同色細線畫初始 matching circle。
- 若 `label_track_id=true`，足球 label 顯示格式為：
  - `football #3 0.91`
  - 非 tracking 目標保持原格式：`goal 0.88`
- `predictions.jsonl` 在 tracking enabled 時新增欄位：
  - `track_id`
  - `track_anchor_x`
  - `track_anchor_y`
  - `track_radius_pixels`
  - `track_first_frame_index`
  - `track_last_seen_frame_index`
  - `track_age_frames`
- 非 target prediction 的上述欄位值為 `null`。

驗收：

- `save_predictions_jsonl=true` 時可看到 tracking 欄位。
- `tracking.enabled=false` 時 JSONL 不強制增加 tracking 欄位，避免破壞既有輸出。
- render class filter 不影響 tracker target filter：即使只 render football，tracking 仍依 `tracking.target_*` 判斷；即使 render all，只有 target football 有 track ID。

## Module 5: Tests

目標：不靠真模型，先用 fake predictions 驗證 tracking 行為。

測試位置：

- 優先在 `tests/test_rf_detr_inference.py` 增加 inference-level 測試。
- 若 tracking helper 測試較多，可新增 `tests/test_rf_detr_video_tracking.py`。

必要測試：

- `test_tracking_same_ball_inside_initial_radius_keeps_track_id`
- `test_tracking_ball_outside_initial_radius_creates_new_track`
- `test_tracking_missing_frames_still_match_initial_circle`
- `test_tracking_multiple_balls_use_nearest_unused_track`
- `test_tracking_ignores_non_target_classes`
- `test_tracking_config_defaults_to_football`
- `test_tracking_disabled_preserves_predictions`
- `test_batched_tracking_replay_matches_one_pass_assignments`
- `test_render_filter_is_separate_from_tracking_filter`

命令：

```bash
uv run python -m unittest tests.test_rf_detr_inference
uv run python -m unittest tests.test_rf_detr_video_tracking
uv run python -m py_compile inference_rf_detr_model.py rf_detr_video_tracking.py
git diff --check
```

## Module 6: Documentation And Example Config

目標：讓使用者知道如何啟用與調整足球跟蹤。

實作要求：

- 在 `README.md` 的 inference 區段加入簡短示例：

```yaml
inference:
  tracking:
    enabled: true
    target_class_names: [football]
    radius_pixels: 80
    draw_trajectory: true
```

- 說明固定初始圓規則：
  - 第一個足球 detection 建立 track。
  - 後續足球中心若落在此 track 初始圓內，視為同一顆球。
  - 若球離開初始圓，會建立新 track。

驗收：

- README 說明與 config 欄位一致。
- 中文/英文 class name 不做硬編碼；實際 class 由 `dataset.class_names` 與 `tracking.target_class_names` 解析。

## 建議執行順序

1. Module 1: Config Schema
2. Module 2: Pure Tracking Helper
3. Module 5: Helper tests for pure tracking
4. Module 3: Video Pipeline Integration
5. Module 4: Rendering And Output Records
6. Module 5: Full inference tests
7. Module 6: Documentation And Example Config

每個 module 完成後都先跑對應測試，再進下一個 module。若某個 module 需要調整公開欄位名稱，必須同步更新 config、README、tests 與本文件。
