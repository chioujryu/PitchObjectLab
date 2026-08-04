# Tiny Soccer Ball 偵測與追蹤：rf_detr_trainer 模組化實作計畫

> **來源**：本文件把研究型 playbook
> [`Single-Camera, Gimbal-Mounted Detection & Tracking of a Persistently Tiny Soccer Ball on a Non-Standard Pickup Field.md`](./Single-Camera,%20Gimbal-Mounted%20Detection%20&%20Tracking%20of%20a%20Persistently%20Tiny%20Soccer%20Ball%20on%20a%20Non-Standard%20Pickup%20Field.md)
> 改寫成「一個一個模組」、可由 Claude Code / Codex 逐步執行的實作清單。
>
> **新增功能**：在 inference 加入「圓形搜尋範圍」足球追蹤（含影片軌跡）。詳見 Phase 1。
>
> **與既有文件的關係**：本文件**取代並修正**
> [`football_tracking_implementation_plan.md`](./football_tracking_implementation_plan.md)。
> 既有文件把搜尋圓「固定在第一次偵測位置、永不更新」；依需求 #2，搜尋圓必須在**每次成功匹配時重新置中**。
> 既有文件**保留不動**（AGENTS.md 規則 9：保留使用者建立的檔案），僅在此標記為 superseded。

---

## 0. 這份文件怎麼用

- 每個 **Module 都是一次獨立的 Claude Code / Codex 任務**：實作 → 跑該 Module 的「測試指令」→ 通過後再進下一個 Module。
- 每個 Module 都有固定區塊：**目標 / 影響檔案 / 實作步驟 / config keys / 驗收 / 測試指令**。
- 依賴順序見「模組總覽」。Phase 1 是使用者明確要求的新功能，請**先做完整個 Phase 1**；Phase 2–5 為 playbook 其餘內容，可依需要挑選。
- 標記說明：`now` = 現在可在 rf_detr_trainer 內完成；`optional` = 可做但非必要；`future` = 超出目前單一偵測器範圍，需另開子專案/repo，本文件只給指引。

## 共同規則（每個 Module 的驗收都必須遵守，摘自 `AGENTS.md`）

1. **Entrypoint 分離**：train / test / inference 三個 entrypoint 維持獨立 Python 檔案；共用邏輯放在新的 helper module，不要讓 standalone entrypoint 互相 import（規則 15）。本計畫只動 inference 行為。
2. **Config 分組**：execution config 一律放 `config/`，分組維持 `runtime / model / dataset / output / inference / evaluation`；新增的每個 key 都要有 `#` 註解寫明用途與合法值（規則 2）。
3. **輸出前估算 + 確認**：任何寫出 images / videos / json / jsonl / config snapshot 之前，先印出檔案數與磁碟用量估算，並要求確認，除非 `--yes` 或 config 略過（規則 4）。
4. **Config snapshot + 計時**：任何寫出 output 的流程都要把產生該 output 的 config 存進 output 資料夾，並寫 `run_timing.json`（HH:MM:SS）（規則 3、19）。
5. **路徑**：不存在的輸出路徑用 `mkdir(parents=True, exist_ok=True)`；同時支援 Windows 與 Linux 路徑（規則 5、7）。
6. **編碼**：UTF-8，保留繁/簡中文（規則 3）。
7. **終端體驗**：entrypoint `import colorama` 並呼叫 `colorama.init(autoreset=True)`；長迴圈用 `tqdm` 進度條（規則 6）。
8. **顏色一致**：同一個 class ID 的 bbox 顏色在整次 run 內一致，沿用既有 `class_color()`（規則 16）。
9. **相依套件**：原則上盡量不新增重相依；`circle` 追蹤器與 `rf_detr_video_tracking.py` 維持**純標準庫**（不載入模型，可離線單元測試）。**例外（依使用者明確要求）**：Module 3.1b 為 inference 接入 OC-SORT / Deep OC-SORT / BoT-SORT / ByteTrack，新增 `boxmot==13.0.0`（連帶 `lapx`/`filterpy`/`gdown`/`pandas`/`scikit-learn` 等，並會把 `numpy` 收斂到 1.26.x）。boxmot 採**延遲匯入**（只在選用 boxmot 演算法時才需要），且不安裝 `[yolo]` extra。WASB / SAM2 / 超解析仍屬 Phase 5。
10. **下載走鏡像**：若某 Module 需要下載模型/資料，先判斷公網 IP 區域，CN/HK/MO/TW 用中國友善鏡像，其餘用官方來源（規則 8）。

## 與既有程式碼的接點（Phase 1 會用到）

| 物件 | 位置 | 用途 |
|---|---|---|
| `draw_predictions()` | `inference_rf_detr_model.py:535` | 影格上畫 bbox/label，Phase 1.4 在此疊加軌跡與圓 |
| `class_color()` | `inference_rf_detr_model.py:510` | 穩定的 per-class 顏色 |
| `predict_video_file_one_pass()` | `inference_rf_detr_model.py`（約 684–775） | 單趟影片推理（逐格） |
| `predict_video_file_batched()` | `inference_rf_detr_model.py`（約 778–923） | 兩趟：偵測 pass + render pass |
| `predictions.jsonl` 寫入 | `inference_rf_detr_model.py` | Phase 1.5 追加 tracking 欄位 |
| `sahi:` config 區塊 | `config/rf_detr_inference.yaml:125` | Phase 2.1 重用 |
| 類別對應 | `config/rf_detr_inference.yaml:51`（`standing_player=0, football=1, goal=2`） | tracking 目標為 `football` |

---

## 模組總覽與相依順序

```
Phase 1（新功能，先全部做完）
  1.1 Config Schema ──► 1.2 Pure Tracking Helper ──► 1.6a Helper 單元測試
                                   │
                                   ▼
                        1.3 Video Pipeline Integration ──► 1.4 Rendering ──► 1.5 Output Records ──► 1.6b Inference 測試
Phase 2（偵測品質，可獨立於 Phase 1）
  2.1 SAHI football preset      2.2 P2/高解析訓練(gated)   2.3 Tiny-object augmentation   2.4 Super-resolution(optional)
Phase 3（多球與主球）            需要 Phase 1 的 track 結構
  3.1 列舉所有球(multi-track)  ──► 3.2 主球選擇            3.3 Camera-motion compensation(optional)
Phase 4（場地邊界，optional）    Phase 5（advanced / future，pointers only）
```

- [x] **1.1** Tracking Config Schema — `now` ✅ 已實作（`config/rf_detr_inference.yaml` 的 `inference.tracking`）
- [x] **1.2** Pure Tracking Helper — `now` ✅ 已實作（`rf_detr_video_tracking.py`）
- [x] **1.3** Video Pipeline Integration — `now` ✅ 已實作（one-pass + batched，`track_id` 一致）
- [x] **1.4** Trajectory & Overlay Rendering — `now` ✅ 已實作（per-track 顏色、taper、目前球心、搜尋圓）
- [x] **1.5** Output Records & Summary — `now` ✅ 已實作（JSONL tracking 欄位 + `tracking_summary.json`）
- [x] **1.6** Tests — `now` ✅ 已實作（`tests/test_rf_detr_video_tracking.py`，含合成影片端到端測試）

> **Phase 1 實作備註**：實際版本在原規格上加了 4 項優化——速度預測閘門（`use_velocity_prediction`，預設關閉以忠於原始規格）、多球軌跡視覺強化（`trajectory_per_track_color`/`trajectory_taper`/`draw_current_center`）、`min_hits` 假軌跡抑制、合成影片端到端測試。
> **第二輪優化（O5–O8）**：速度閘門強化（靜止 ∪ 預測雙圓 + `velocity_smoothing` EMA，接住快速移動後突然停住的球）、全域最近鄰關聯（多球交會時減少 `track_id` 互換）、軌跡記憶體上限（`deque`，長影片不爆記憶體）、追蹤 CLI 開關（`--track`/`--no-track`/`--track-radius`/`--track-velocity`）。
> **軌跡消失（render-only）**：`trajectory_max_age_frames`（預設 30）讓軌跡只畫最近 N 影格的點，球離開畫面後軌跡逐點縮回並消失、不再永久殘留；不影響 `track_id` 與 `tracking_summary.json`。
> **斷斷續續修正（track 衛生 + 斷點補插）**：`max_missing_frames` 預設改為有限 `30`（過期清除 stale track，避免幾十條永不消失的 track 讓真球在它們之間跳號）、`min_hits` 提高可濾掉單格雜訊 track、`use_velocity_prediction: true` 讓 gate 跟著球；渲染端在偵測斷點用速度外推 live 位置（`live_center`），讓球與軌跡頭不會卡住凍結。完整 config 欄位以 `config/rf_detr_inference.yaml` 為準。
- [ ] **2.1** SAHI football preset — `now`
- [ ] **2.2** P2 / 高解析訓練 preset — `now (gated)`
- [ ] **2.3** Tiny-object augmentation preset — `now`
- [ ] **2.4** Super-resolution ROI — `optional`
- [ ] **3.1** 列舉所有球（multi-track）— `now`
- [x] **3.1b** OC-SORT / Deep OC-SORT / BoT-SORT / ByteTrack（boxmot 介接）— `now` ✅ 已實作（`inference.tracking.algorithm`，預設 `circle`；各 boxmot 演算法用巢狀區塊）
- [ ] **3.2** 主球選擇（player proximity + 群聚 + 遲滯）— `now`
- [ ] **3.3** Camera-motion compensation — `optional`
- [ ] **4.1** HSV grass ∩ player hull active region — `optional`
- [ ] **4.2** Learned / SAM2 segmentation — `future`
- [ ] **5.x** WASB / SAMURAI / 離線管線 / 資料蒐集 / 多 GPU — `future`

---

# Phase 1 — Inference 足球圓形追蹤（新功能，最高優先）

## 追蹤演算法規格（Module 1.2–1.4 共同依據）

> 這是新功能的核心，先把規則定死，後續 Module 不得重新詮釋。

**名詞**：`center` = bbox 中心 `(x + w/2, y + h/2)`；`dist` = 兩中心的歐式距離；目標類別預設 `football`。

每條 track（`TrackedBall`）保存：`track_id`、`center`（**目前**搜尋圓圓心）、`base_radius`、`missing_frames`、`first_frame_index`、`last_seen_frame_index`、`points: list[(frame_index, x, y)]`。

**有效半徑**（處理快球 / 漏檢後重現）：

```
base_radius   = radius_pixels
              若 radius_scale 非 null： max(radius_pixels, radius_scale * max(bbox_w, bbox_h))
effective_r   = base_radius + radius_growth_per_missing_frame * missing_frames
              若 max_radius_pixels 非 null： min(effective_r, max_radius_pixels)
```

**匹配判定**：detection 屬於某 track 的條件是 `dist(det.center, track.center) <= effective_r`。
**絕不使用 IoU / 面積重疊**——快球相鄰影格 bbox 可能完全不重疊（需求 #5），只能用「中心距離 vs 半徑」。

**`FootballTracker.update(frame_index, predictions) -> list[dict]`**（純函式，不載入模型、不修改輸入物件、回傳順序與輸入一致）：

```text
1. targets ← 只取目標類別的 predictions，依 score 由高到低排序
2. used ← ∅；assignment ← {}                      # prediction index → TrackedBall
3. for det in targets（高分先選）:
       cand ← 所有「未在 used、且 dist(det.center, tr.center) <= effective_r(tr)」的 track
       若 cand 非空：
           best ← cand 中 dist 最小者；平手取較小 track_id
           best.center            ← det.center      # ★ 重新置中（需求 #2「確認後圓心更新」）
           best.base_radius       ← base_radius(det) # 圓半徑隨球的視覺大小刷新
           best.missing_frames    ← 0
           best.last_seen_frame_index ← frame_index
           best.points.append((frame_index, det.center))
           used.add(best.track_id)；assignment[det] ← best
       否則：
           tr ← 新建 track（新 track_id、center=det.center、base_radius、missing_frames=0、points=[該點]）
           assignment[det] ← tr
4. for tr in tracks 且 tr.track_id ∉ used:          # 本格沒匹配到的 track
       tr.missing_frames += 1                        # 圓心停在上次位置（需求 #2「相隔數格仍用既有圓」）
5. 若 max_missing_frames 是整數：移除 missing_frames > max_missing_frames 的 track
   （預設 all/None：永不過期）
6. 回傳：對每個原始 prediction，目標類別附上 assignment 的 tracking 欄位；非目標類別 tracking 欄位為 None
```

**需求對應檢核表**：

| 需求 | 對應設計 |
|---|---|
| #1 inference 加追蹤 | `inference.tracking.enabled`；Phase 1.3 接入影片管線 |
| #2 偵測 → 往外擴圓 | 新 track `center = det.center`、半徑 `base_radius` |
| #2 下一格在圓內 = 同球 | `dist(det.center, track.center) <= effective_r` |
| #2 相隔數格仍用既有圓 | 未匹配 track 不刪除（預設 `max_missing_frames: all`），圓心停在上次位置 |
| #2 確認後圓心更新到最新球心 | 匹配成功時 `best.center ← det.center` |
| #3 video 追蹤有軌跡 | 每 track 累積 `points`；Phase 1.4 畫 polyline |
| #5 快球、bbox 不重疊 | 只用中心距離；`radius_growth_per_missing_frame` + 夠大的 `radius_pixels/radius_scale` 讓遠跳的球仍落在圓內 |

多球自然支援（一球一圓），對應 playbook §B。

---

## Module 1.1 — Tracking Config Schema

**目標**：先讓 config 描述追蹤行為，但**不改任何推理邏輯**。

**影響檔案**：`config/rf_detr_inference.yaml`（必要）；任何專用 inference video config（若存在，同步加上同樣區塊）。

**實作步驟**：在 `inference:` 之下新增 `tracking:` 區塊：

```yaml
inference:
  # ...（既有 keys 不動）...
  tracking:
    # Enable football tracking for video inference. Options: true, false.
    enabled: false
    # Category IDs to track. Empty means use target_class_names. Options: list of category IDs.
    target_class_ids: []
    # Category names to track. Empty with no target_class_ids defaults to football. Options: names such as [football].
    target_class_names:
      - football
    # Base search radius (pixels) around a ball center for same-ball matching. Options: positive number.
    radius_pixels: 80
    # Optional size-relative base radius: max(radius_pixels, radius_scale * max(bbox_w, bbox_h)). Options: null to disable, or positive number.
    radius_scale: null
    # Radius growth per missing detection frame, to catch fast/returning balls after gaps. Options: non-negative number.
    radius_growth_per_missing_frame: 0.0
    # Hard cap on the grown effective radius (pixels). Options: null for no cap, or positive number.
    max_radius_pixels: null
    # Frames a track may stay unmatched before it expires. Options: all/null to never expire, or a non-negative integer.
    max_missing_frames: all
    # Confirmations before a track is treated as stable (reserved for main-ball selection). Options: positive integer.
    min_hits: 1
    # Draw trajectory polyline for tracked balls in rendered videos. Options: true, false.
    draw_trajectory: true
    # Trajectory length in points/frames. Options: null for full history, or a positive integer.
    trajectory_max_points: 30
    # Trajectory line width (pixels). Options: positive integer.
    trajectory_width: 2
    # Draw the current search circle for each active track (debug overlay). Options: true, false.
    draw_search_circle: false
    # Include the track ID in rendered football labels, e.g. "football #3 0.91". Options: true, false.
    label_track_id: true
```

- 第一版**不新增 CLI override**，tracking 只由 YAML 控制。
- `tracking` 區塊缺漏時要能用上述預設值（在 Phase 1.2 的 `parse_tracking_config` 補預設）。

**驗收**：
- `yaml.safe_load()` 可解析所有更新後的 inference config。
- `enabled: false` 時，inference 行為與輸出**完全不變**。

**測試指令**：
```bash
uv run python -c "import yaml,glob; [yaml.safe_load(open(p,encoding='utf-8')) for p in glob.glob('config/*inference*.yaml')]; print('ok')"
```

---

## Module 1.2 — Pure Tracking Helper

**目標**：新增純邏輯 helper，先**不接入**影片管線，可單獨單元測試。

**影響檔案**：新增 `rf_detr_video_tracking.py`（頂部 `from __future__ import annotations`）。

**實作步驟**：

- 資料結構（用 `@dataclass`）：
  - `TrackingConfig`：`enabled, target_class_ids: set[int], radius_pixels, radius_scale, radius_growth_per_missing_frame, max_radius_pixels, max_missing_frames: Optional[int], min_hits, draw_trajectory, trajectory_max_points: Optional[int], trajectory_width, draw_search_circle, label_track_id`。
  - `TrackedBall`：`track_id, center_x, center_y, base_radius, missing_frames, first_frame_index, last_seen_frame_index, points: list[tuple[int, float, float]]`。
- 函式：
  - `parse_tracking_config(config, categories) -> TrackingConfig`：
    - 從 `inference.tracking` 讀值並補預設。
    - 目標類別解析：`target_class_ids` 優先；空則用 `target_class_names`（對 `categories` 名稱做 casefold 比對）；兩者皆空 → fallback `football`。
    - 驗證：`radius_pixels > 0`、`trajectory_width > 0`、`max_missing_frames` 為 `all/null` 或非負整數；非法值丟出明確錯誤訊息。
  - `bbox_center(prediction) -> tuple[float, float]`：由 COCO `xywh` 算中心。
  - `effective_radius(track, cfg) -> float`：依「追蹤演算法規格」公式。
  - `FootballTracker`（持有 `cfg`、`tracks: list[TrackedBall]`、遞增的 `next_id`）：
    - `update(frame_index, predictions) -> list[dict]`：完全照「追蹤演算法規格」的 1–6 步。
    - 不修改輸入；目標 prediction 附 `track_id, track_center_x, track_center_y, track_radius_pixels, track_first_frame_index, track_last_seen_frame_index, track_age_frames`；非目標附同名欄位但值為 `None`。
- 純 Python + 標準庫即可（距離用 `math.hypot`）；**不得**載入 RF-DETR。

**驗收**：見 Module 1.6a 單元測試全數通過；`enabled=false` 時呼叫端可直接略過 tracker（helper 本身不假設一定啟用）。

**測試指令**：
```bash
uv run python -m py_compile rf_detr_video_tracking.py
```

---

## Module 1.3 — Video Pipeline Integration

**目標**：讓 one-pass 與 batched 兩種影片推理用同一套 tracking 邏輯，且在相同偵測序列下產生**一致的 `track_id`**。

**影響檔案**：`inference_rf_detr_model.py`。

**實作步驟**：
- 在影片推理開始時，用 `parse_tracking_config()` 建 `TrackingConfig`；`enabled=false` 時**不建立** tracker，且不改變既有流程/速度/輸出。
- `predict_video_file_one_pass()`：
  - 僅在 detection frame（`should_detect=true`）呼叫 `tracker.update(absolute_frame_index, frame_predictions)`。
  - 寫入 `all_predictions` 的是 tracker 更新後的 rows。
  - skipped frame 沿用 `last_predictions` 與 tracker 既有 `points` 來渲染（軌跡持續顯示）。
- `predict_video_file_batched()`：
  - detection pass 只快取 **raw** predictions（依 `segment_frame_index`）。
  - render pass **依影格順序** replay：遇 detection frame 才呼叫 `tracker.update(...)`，再用更新後 predictions 渲染；replay 後才寫 `all_predictions`，確保 JSONL 與畫面順序一致。
- 兩種模式在相同 prediction 序列下 `track_id` 必須一致（靠相同的逐格 `update` 呼叫順序保證）。

**驗收**：
- 既有 `inference.video.batch_size` 等行為保留。
- `render_skipped_frames=true`：skipped frame 看得到最新 bbox 與累積軌跡。
- `render_skipped_frames=false`：只有 detection frame 被畫 bbox/軌跡。
- `enabled=false`：輸出與目前**逐位元組相同**。

**測試指令**：見 Module 1.6b（`test_batched_tracking_replay_matches_one_pass_assignments`）。

---

## Module 1.4 — Trajectory & Overlay Rendering

**目標**：在影片上畫足球軌跡（需求 #3），並可選擇畫出目前搜尋圓。

**影響檔案**：`inference_rf_detr_model.py`（`draw_predictions()`，:535）。

**實作步驟**：
- 擴充 `draw_predictions()`，新增**選用**參數（例如 `tracks=None, tracking_cfg=None`）；不傳時行為與現在一致。
- 顏色：bbox 仍用 `class_color(category_id)`；軌跡線預設用 football class color（或穩定的 per-track 色，二擇一並在註解說明）。
- `draw_trajectory=true`：對每條 active track，取最後 `trajectory_max_points` 個點，用 `ImageDraw.line` 畫 polyline，線寬 `trajectory_width`。
- `draw_search_circle=true`：用同色細線畫該 track 目前的搜尋圓（圓心 `center`、半徑 `effective_radius`）。
- `label_track_id=true`：足球 label 顯示為 `football #3 0.91`；非追蹤目標維持原格式 `goal 0.88`。
- 維持既有 PIL→BGR 流程（畫在 RGB canvas，再由呼叫端轉回 BGR 寫入 `cv2.VideoWriter`）。

**驗收**：
- 影片可見每顆被追蹤足球的連續軌跡；`track_id` 穩定不亂跳（同一顆球不換號）。
- 關掉 `draw_trajectory`/`draw_search_circle`/`label_track_id` 各自獨立生效。

**測試指令**：以 Phase 1.6 的 fake-frame 測試 + 人工檢視一段樣本影片輸出（見 Appendix C）。

---

## Module 1.5 — Output Records & Summary

**目標**：把 tracking 結果寫進 prediction records，並輸出追蹤摘要。

**影響檔案**：`inference_rf_detr_model.py`（JSONL 與 summary 寫出處）。

**實作步驟**：
- `save_predictions_jsonl=true` 且 tracking enabled 時，於 `predictions.jsonl` 每筆**目標** prediction 追加：
  `track_id, track_center_x, track_center_y, track_radius_pixels, track_first_frame_index, track_last_seen_frame_index, track_age_frames`。非目標 prediction 上述欄位為 `null`。
- 新增 `tracking_summary.json`：每條 track 的 `track_id, first_frame_index, last_seen_frame_index, num_points, lifespan_frames`，以及總 track 數。
- 沿用既有 output estimate / confirmation / config snapshot / `run_timing.json` 規則（共同規則 3、4）。

**驗收**：
- `enabled=true`：JSONL 出現 tracking 欄位，且 `tracking_summary.json` 內容與畫面一致。
- `enabled=false`：JSONL schema 不變（不強制新增欄位），避免破壞既有輸出。
- render class filter 與 tracker target filter 互不影響：只 render football 時 tracking 仍依 `tracking.target_*` 判斷；render all 時也只有目標 football 有 `track_id`。

**測試指令**：見 Module 1.6b。

---

## Module 1.6 — Tests

**目標**：不靠真模型，用 fake predictions 驗證追蹤行為。

**影響檔案**：新增 `tests/test_rf_detr_video_tracking.py`；必要時於 `tests/test_rf_detr_inference.py` 加 inference-level 測試。

**必要測試**：
- `test_same_ball_inside_radius_keeps_track_id`
- `test_recenter_follows_moving_ball`（連續小位移，圓心跟著移動）
- `test_missing_frames_then_rematch_keeps_track_id`（中間數格漏檢，球回到既有圓內 → 同 id）
- `test_radius_growth_catches_far_jump`（快球遠跳、bbox 不重疊，靠成長半徑仍匹配 → 需求 #5）
- `test_ball_outside_radius_creates_new_track`
- `test_multiple_balls_use_nearest_unused_track`
- `test_ignores_non_target_classes`
- `test_config_defaults_to_football`
- `test_disabled_preserves_predictions`
- `test_batched_tracking_replay_matches_one_pass_assignments`
- `test_render_filter_is_separate_from_tracking_filter`

**測試指令**：
```bash
uv run python -m unittest tests.test_rf_detr_video_tracking
uv run python -m unittest tests.test_rf_detr_inference
uv run python -m py_compile inference_rf_detr_model.py rf_detr_video_tracking.py
git diff --check
```

---

# Phase 2 — Tiny-ball 偵測品質（playbook §A）

> 目的：在追蹤之前/之外，先把「持續只有 <10–15px 的球」的偵測 recall 拉高。對應 playbook §A。

## Module 2.1 — SAHI 足球 inference preset — `now`

**目標**：用既有 SAHI（`config/rf_detr_inference.yaml:125` 的 `sahi:`、`inference.mode: sahi`）做一組對小球友善的 preset。
**影響檔案**：新增 `config/rf_detr_inference_sahi_football.yaml`（複製現有 inference config 再調 `sahi.slice_*`、`overlap_*`、`recheck`）。
**實作步驟**：`mode: sahi`；縮小 slice（例如 512/640）、`overlap 0.2–0.3`、`postprocess GREEDYNMM + IOS`（規則 10）；開 `sahi.recheck` 對 football 做二次中心裁切驗證（規則 14）。
**驗收**：在樣本影像/影片上，football recall 較 `full_image` 提升；無重複框（GREEDYNMM+IOS 生效）。
**測試指令**：`uv run python inference_rf_detr_model.py --config config/rf_detr_inference_sahi_football.yaml --dry-run`（先看估算），再小量 `--yes` 實跑比較。

## Module 2.2 — P2 high-res head / 高解析訓練 preset — `now (gated)`

**目標**：playbook §A3，加 P2（stride 4）head 或提高輸入解析度以涵蓋 ~4×4px 目標。
**影響檔案**：`config/rf_detr_train.yaml`（新增 high-res / P2 preset）。
**Gate（先驗證）**：先確認 `rfdetr` 套件是否支援 P2 head 或自訂 `model.resolution`/backbone stride。
- 若支援 → 建立訓練 preset（提高 resolution、必要時調整 head）。
- 若不支援 → 在本 Module 標記為 `future`，記錄需改的 upstream 點，不硬改。
**驗收**：preset 可啟動訓練（`--dry-run` 通過）；或明確記錄不支援的結論。
**測試指令**：`uv run python train_rf_detr_model.py --config <preset> --dry-run`。

## Module 2.3 — Tiny-object augmentation preset — `now`

**目標**：playbook §H，對小球做 copy-paste / mosaic / 高解析訓練增強。
**影響檔案**：`config/rf_detr_train.yaml`（augmentation 區段，依 rfdetr 支援的 aug 參數）。
**實作步驟**：開啟/加大 mosaic、small-object copy-paste（若支援）、提高輸入解析度；其餘維持。
**驗收**：訓練可啟動且 augmentation 生效（dataset grid 取樣可見增強效果）。
**測試指令**：`uv run python train_rf_detr_model.py --config <preset> --demo --yes`（demo 小量）。

## Module 2.4 — Super-resolution 場地 ROI（offline）— `optional`

**目標**：playbook §A4，對 field ROI 做超解析後再偵測。屬較重流程，預設 `optional`。
**影響檔案**：新增離線前處理腳本（不動三個 entrypoint）。
**備註**：需評估新套件/模型下載（走鏡像，規則 8）。若不做，保留為 `future`。

---

# Phase 3 — 多球與主球選擇（playbook §B、§C）

## Module 3.1 — 列舉所有球（multi-track）— `now`

**目標**：playbook §B1/§B2。Phase 1 的 `FootballTracker` 已天然支援多球（一球一圓），本 Module 確認低門檻偵測 + 多 track 輸出穩定。
**影響檔案**：`config/rf_detr_inference.yaml`（必要時降低 `model.confidence_threshold` 以保 recall）；`rf_detr_video_tracking.py`（確認多 track 行為）。
**驗收**：多顆球同框時各自有獨立 `track_id`，交會後不長期黏錯（短暫 ID switch 可接受，由 3.2 再選主球）。

## Module 3.1b — OC-SORT / Deep OC-SORT / BoT-SORT / ByteTrack（boxmot 介接）— `now` ✅ 已實作

**目標**：playbook §B2。把成熟的多物件追蹤器（含小/快物件質心關聯與相機運動補償）接進 inference，讓使用者可在 `circle` 與 boxmot 系列之間切換；預設 `circle`。追蹤類別可設定，預設足球（重用既有 `target_class_ids`/`target_class_names`）。

**影響檔案**：
- `config/rf_detr_inference.yaml`：`inference.tracking` 新增 `algorithm` 與 boxmot 參數（每個 key 附 `#` 註解，規則 2）。
- `rf_detr_video_tracking.py`：`TrackingConfig` / `parse_tracking_config` 加入 `algorithm` 與 boxmot 欄位與驗證；`FootballTracker.update` 加 `frame=None`（circle 忽略）。本檔**維持純標準庫**。
- `rf_detr_boxmot_tracker.py`（新增）：`BoxmotTracker` 介接層，**延遲匯入** boxmot。
- `inference_rf_detr_model.py`：`create_tracker` 工廠 + `resolved_tracker_device`；兩處 tracker 建構改用工廠、兩處 `tracker.update(..., frame=frame)` 傳入影格；新增 `--tracker`/`--reid-weights` CLI 與 ReID 區域鏡像警示（重用 `scripts/setup_pytorch_uv.py:detect_region`）。
- `tests/test_rf_detr_boxmot_tracker.py`（新增）：以假 boxmot tracker 離線驗證。

**介接層演算法規格**：
- 只把目標類別的 prediction 轉成 boxmot 的 `dets`（N×6 `[x1,y1,x2,y2,conf,cls]`，COCO xywh→xyxy）。
- 呼叫 `tracker.update(dets, frame)` 取得 M×8 `[x1,y1,x2,y2,track_id,conf,cls,det_ind]`，用 **`det_ind`** 把 track 對回原始 prediction（M 可能 < N 且亂序，**不可用位置對應**）。
- 為每個 boxmot track id 維護一個 `TrackedBall`（圓心＝回傳 bbox 中心、`points` 軌跡、`hits`、`missing_frames`），因此既有渲染（`draw_track_overlays`）、`predictions.jsonl` 追蹤欄位、`tracking_summary.json` 完全沿用、不需更動。

**config keys**：`algorithm`（`circle`/`ocsort`/`deepocsort`/`botsort`/`bytetrack`，預設 `circle`）、top-level 共用 `reid_weights`/`reid_device`/`reid_half`/`cmc_method`/`per_class`，以及每個 boxmot 演算法各自的**巢狀區塊** `ocsort:`/`deepocsort:`/`botsort:`/`bytetrack:`（完整清單見 Appendix B）。parser 把巢狀區塊對映到扁平的 `TrackingConfig` 欄位，adapter 不受影響。

**相依（規則 8）**：`uv add 'boxmot==13.0.0'`（海外用官方 index；CN/HK/MO/TW 用 Tsinghua）；不裝 `[yolo]` extra。需驗證 `torch==2.11.0`/`torchvision==0.26.0` 未被動到（boxmot 會把 `numpy` 收斂到 1.26.x，實測 torch/torchvision/cv2/boxmot 共存正常）。

**ReID（規則 8）**：`ocsort`/`bytetrack` 不需權重；`deepocsort`/`botsort` 用外觀 ReID，預設 `osnet_x0_25_msmt17.pt` 走 Google Drive 自動下載（CN 不穩）。請設 `reid_weights` 本機路徑略過下載，或用 `botsort.with_reid: false` / `algorithm: ocsort`。`reid_half` 僅 GPU 有效，CPU/MPS 自動關閉。

**驗收**：
- `algorithm: circle`（或任何演算法 `enabled: false`）輸出與既有版本逐位元組一致；既有 circle 測試全綠。
- boxmot 系列：`det_ind` 對回正確、非目標類別追蹤欄位為 `None`、`.tracks` 可被既有渲染 helper 消費。
- boxmot 未安裝卻選了 boxmot 演算法 → **執行期**丟出含安裝指引的清楚 ImportError（匯入期不報錯）。

**測試指令**：
```bash
uv run python -m unittest tests.test_rf_detr_boxmot_tracker
uv run python -m unittest tests.test_rf_detr_video_tracking
uv run python -m py_compile inference_rf_detr_model.py rf_detr_video_tracking.py rf_detr_boxmot_tracker.py
# 端到端（OC-SORT，無下載）：先 --dry-run 看估算，再去掉 --dry-run、加 --yes 實跑
uv run python inference_rf_detr_model.py --config config/rf_detr_inference.yaml \
  --source ocsort --output-dir runs/rf_detr/inference_ocsort_demo --dry-run < sample_video.mp4 > --tracker
```

## Module 3.2 — 主球選擇（player proximity + 群聚 + 遲滯）— `now`

**目標**：playbook §C。從 N 條球 track 選出「比賽用球」。
**影響檔案**：新增 helper（例如 `rf_detr_main_ball.py`）；`config/rf_detr_inference.yaml` 新增 `inference.main_ball` 區塊；render 標示主球。
**實作步驟**：
- 用 player（class `standing_player=0`）位置算「動作中心」：最密群聚中心（DBSCAN，若引入 `scikit-learn` 需先評估，否則自寫簡化群聚）＋ player 凸包/質心。
- 每條球 track 評分：到最近 player 腳部與動作中心的距離（近者高分）＋ 持球環（possession ring）＋ track 品質。
- 時間穩定：遲滯（competitor 要連續 K 格超過 incumbent 才換）＋ 分數 EMA；輸出每格 `main_ball` 旗標。
**驗收**：單一比賽球在連續影格被穩定標為主球，邊緣/鄰場球不被誤選；切換有遲滯、不閃爍。
**備註**：若引入 `scikit-learn`，需符合規則 8/9（先評估、走鏡像、別做無關重構）。

## Module 3.3 — Camera-motion compensation（平移雲台）— `optional`

**目標**：playbook §E。雲台持續平移時，先用 ORB+RANSAC affine 估背景運動，匹配前把 track 圓心做位移補償。
**影響檔案**：`rf_detr_video_tracking.py`（在 `update` 前選用補償）；config 加開關。
**備註**：用 OpenCV（已有），不新增重相依。預設關閉。

---

# Phase 4 — Markings-free 場地邊界（playbook §D，optional）

## Module 4.1 — HSV grass ∩ player convex hull — `optional`
**目標**：playbook §D2/§D3。用 HSV 草地遮罩 ∩ 膨脹後的 player 凸包，界定「有效比賽區」，把區外的球（鄰場/邊緣）在主球評分中否決。
**影響檔案**：新增 helper；`inference.main_ball` 加 inside-field 加分/outside-field 否決。
**備註**：純 OpenCV/numpy。

## Module 4.2 — Learned / SAM2 segmentation — `future`
playbook §D2 的學習式分割（DeepLab/SAM2）；重相依，另開子專案。

---

# Phase 5 — Advanced / Future（playbook §A1、§B3、§G、§H；僅指引）

> 以下超出目前單一 RF-DETR 偵測器範圍，需另開子專案/repo。本文件只列指引與 playbook §I 的 repo 連結，不在 rf_detr_trainer 內直接實作。

- **WASB / TrackNet 熱圖球網**（playbook §A1）：多影格熱圖偵測，提升極小球 recall。repo：`nttcom/WASB-SBDT`、`qaz812345/TrackNetV3`。注意 WASB 單球限制。
- **SAMURAI / SAM2 單物件回復**（playbook §B3）：主球確定後鎖定追蹤。repo：`yangchris11/samurai`。
- **離線管線**（playbook §G）：超解析 + 大模型 + RAFT 稠密光流 + GSI 內插 + 雙向/非因果主球選擇 + TrackEval（HOTA/MOTA/IDF1）。
- **資料蒐集與標註**（playbook §H）：實機影片的 domain gap 最大風險；SAM2 預標 + 人工修正。
- **多 GPU / TensorRT 部署**（playbook §F）：4×RTX 5090、FP8/FP4、Triton。

---

# Appendix A — Playbook §A–J → Module 對應

| Playbook 章節 | 對應 Module | 狀態 |
|---|---|---|
| §A 持續極小球偵測（A2 SAHI / A3 P2 / A4 SR / A5 時間聚合） | 2.1 / 2.2 / 2.4 / Phase 1 追蹤連續性 | now / future |
| §A1 熱圖多影格球網（WASB/TrackNet） | Phase 5 | future |
| §B 多球多物件追蹤（B1/B2） | **Phase 1（圓形追蹤＝多 track）**、3.1 | now |
| §B2 攝影機運動補償關聯 | 3.3 | optional |
| §B3 單物件追蹤回復（SAMURAI） | Phase 5 | future |
| §C 主球選擇（球員群聚） | 3.2 | now |
| §D Markings-free 場地邊界 | 4.1 / 4.2 | optional / future |
| §E 攝影機運動補償（雲台） | 3.3（線上）/ Phase 5（RAFT 離線） | optional / future |
| §F 線上管線 | rf_detr_trainer inference 即此路徑（架構參考） | reference |
| §G 離線管線 | Phase 5 | future |
| §H 資料與訓練 | 2.3（增強）/ Phase 5（蒐集標註） | now / future |
| §I 框架與 repo | 各 Module 的連結來源 | reference |
| §J 建議與注意事項 | 併入本文件 caveats | reference |

# Appendix B — `inference.tracking` config key 速查

| Key | 預設 | 說明 |
|---|---|---|
| `enabled` | `false` | 是否啟用影片追蹤 |
| `target_class_ids` / `target_class_names` | `[] / [football]` | 追蹤目標類別 |
| `radius_pixels` | `80` | 基礎搜尋半徑（像素） |
| `radius_scale` | `null` | 尺寸相關半徑 `max(radius_pixels, scale*ball_size)` |
| `radius_growth_per_missing_frame` | `0.0` | 漏檢時半徑成長（接快球/重現） |
| `max_radius_pixels` | `null` | 成長半徑上限 |
| `max_missing_frames` | `30` | 漏檢多少格後過期（有限值避免 stale track 累積、真球跳號）；`all/null` 永不過期 |
| `min_hits` | `1` | 視為穩定 track 的確認次數；只有確認的 track 才畫軌跡/標 ID |
| `use_velocity_prediction` / `velocity_smoothing` | `false / 0.5` | 速度閘門（靜止 ∪ 預測雙圓）與速度 EMA 平滑 |
| `draw_trajectory` / `trajectory_max_points` / `trajectory_width` | `true / 30 / 2` | 軌跡渲染（`trajectory_max_points` 同時是記憶體上限；`null`→1024） |
| `trajectory_max_age_frames` | `30` | 超過這麼多「影格」沒被偵測到，軌跡會逐點縮回並消失（球離開畫面後不殘留）；`null`=永久保留 |
| `trajectory_per_track_color` / `trajectory_taper` / `draw_current_center` | `true / true / true` | 多球軌跡視覺：每球不同色、由舊到新漸粗、標目前球心 |
| `draw_search_circle` | `false` | debug：畫目前搜尋圓（circle 專用） |
| `label_track_id` | `true` | label 顯示 `football #id score` |

**boxmot 介接（Module 3.1b；`algorithm` 非 `circle` 時生效；circle 專用半徑/速度欄位此時忽略）**

| Key | 預設 | 說明 |
|---|---|---|
| `algorithm` | `circle` | 追蹤器：`circle`/`ocsort`/`deepocsort`/`botsort`/`bytetrack` |
| `target_class_ids` / `target_class_names` | `[] / [football]` | 追蹤目標類別（所有演算法共用，預設足球） |
| `reid_weights` | `null` | deepocsort/botsort 外觀 ReID 權重；填本機路徑可略過 Google Drive 下載 |
| `reid_device` | `null` | ReID 裝置；`null` 沿用 `model.device` |
| `reid_half` | `false` | FP16 ReID（僅 GPU；CPU/MPS 自動關閉） |
| `cmc_method` | `ecc` | 相機運動補償：`ecc`/`orb`/`sof`/`null` |
| `per_class` | `false` | 各類別獨立 ID 空間 |
| `ocsort:` (巢狀) | 見 config | OC-SORT：`det_thresh 0.2`/`max_age 30`/`min_hits 3`/`asso_threshold 0.3`/`delta_t 3`/`asso_func iou`/`inertia 0.2`/`use_byte false` |
| `deepocsort:` (巢狀) | 見 config | Deep OC-SORT：`det_thresh 0.3`/`max_age 30`/`min_hits 3`/`iou_threshold 0.3`/`delta_t 3`/`asso_func iou`/`inertia 0.2`/`w_association_emb 0.5`/`alpha_fixed_emb 0.95`/`embedding_off false`/`cmc_off false` |
| `botsort:` (巢狀) | 見 config | BoT-SORT：`track_high_thresh 0.5`/`track_low_thresh 0.1`/`new_track_thresh 0.6`/`track_buffer 30`/`match_thresh 0.8`/`proximity_thresh 0.5`/`appearance_thresh 0.25`/`with_reid true`/`fuse_first_associate false` |
| `bytetrack:` (巢狀) | 見 config | ByteTrack：`track_thresh 0.45`/`match_thresh 0.8`/`track_buffer 25`/`frame_rate 30` |

# Appendix C — 驗證 cheatsheet

```bash
# 1) Phase 1 純邏輯（不需模型）
uv run python -m unittest tests.test_rf_detr_video_tracking

# 2) 語法/編譯與空白檢查
uv run python -m py_compile inference_rf_detr_model.py rf_detr_video_tracking.py
git diff --check

# 3) 端到端：對樣本影片開啟追蹤（先看估算，再實跑）
uv run python inference_rf_detr_model.py \
  --config config/rf_detr_inference.yaml \
  --source \
  runs/rf_detr/inference_tracking_demo \
  --dry-run < sample_video.mp4 > --output-dir
# 編輯 config：inference.tracking.enabled: true，再去掉 --dry-run、加 --yes 實跑

# 人工檢視重點：
#  - 同一顆球的 track_id 穩定不跳號
#  - 影片上有連續軌跡（trail）
#  - 快球遠跳（相鄰格 bbox 不重疊）仍延續同一 track（半徑成長生效）
#  - predictions.jsonl 內含 track_id 等欄位；tracking_summary.json 與畫面一致
#  - tracking.enabled=false 時輸出與既有版本一致
```
