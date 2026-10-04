# SentinelVision — Project Context for AI Agents

**Purpose:** Give future contributors a reliable, source-grounded map of the repository and its current behavior. Read this before making architectural or API changes.

**Last reviewed:** 2026-10-03
**Repository root:** `D:\SentinalVision` (Windows)
**Naming note:** The folder is spelled `SentinalVision`; the project is named SentinelVision.

## 1. Project at a glance

SentinelVision is a real-time multimodal surveillance **research prototype**. It combines video detections and acoustic classifications into a heuristic event-risk assessment. The root implementation is the source of truth; older reports and guides sometimes describe APIs or features that are no longer present.

The current baseline is:

- **Vision:** Ultralytics YOLO for generic objects/persons plus OpenCV Haar face detection.
- **Audio:** Microphone capture plus a Hugging Face Audio Spectrogram Transformer (AST) model.
- **Fusion:** Deterministic weighted evidence, limited crowd/face-ratio context, EMA smoothing, risk labels, and alert cooldown.
- **Runtime:** Camera frames are processed synchronously; audio capture/inference runs in the background.
- **Research status:** No trained weapon detector, validated gunshot detector, object tracker, learned cross-modal attention, or LSTM temporal model is implemented.

Scores are model/heuristic outputs, **not calibrated probabilities or proof of an incident**. The system must not be described as a validated safety, threat, or weapon-recognition system. Use controlled, staged data collection and document consent, scenario, labels, and evaluation protocol.

## 2. Source of truth and repository map

Prefer current root source and tests over prose claims in older documents. The `backup_stabilization_20260902_051149/` and `SentinelVision_Stabilization_Patch/` directories are historical copies/artifacts, not the active implementation. `venv/`, model weights, benchmark outputs, and local media are environment/data artifacts, not source modules.

| Path | Role |
|---|---|
| [audio.py](audio.py) | `AudioDetectorConfig`, audio stream, rolling buffer, AST inference, threat weighting, health, and shutdown. |
| [vision.py](vision.py) | `VisionDetectorConfig`, YOLO/Haar processing, timing, cached face counts, and health. |
| [fusion_engine.py](fusion_engine.py) | Authoritative fusion policy: `FusionConfig`, feature/result dataclasses, contextual vision score, sensor renormalization, thresholds, EMA. |
| [sentinel_vision.py](sentinel_vision.py) | `SentinelVisionUnified` orchestrator, result schema, sensor freshness/availability, reasoning, alerts, status. |
| [demo_sentinel_unified.py](demo_sentinel_unified.py) | Camera demo; current CLI only accepts `--camera`; `q` quits. |
| [quickstart_unified.py](quickstart_unified.py) | Minimal unified pipeline example; use it as a real usage reference alongside tests. |
| [benchmark.py](benchmark.py) | Camera benchmark/CSV and JSON output; currently has a result-schema mismatch (see Known issues). |
| [collect_dataset.py](collect_dataset.py) | Manual image/audio/metadata capture for controlled scenarios. |
| [test_fusion_engine.py](test_fusion_engine.py) | Deterministic tests for fusion/scoring/availability. |
| [test_sentinel_unified.py](test_sentinel_unified.py) | Orchestrator tests with mocked sensor constructors. |
| [test_audio.py](test_audio.py) | Currently empty; does not provide audio test coverage. |
| [demo/app.py](demo/app.py) | Separate Streamlit research dashboard for one-frame YOLO, saved M2 audio baseline, and research artifacts; not the unified live pipeline. |
| [research/](research/) | Dataset manifest/split scripts and M2 audio baseline feature/training/analysis workflow. |
| [research_data/protocol/](research_data/protocol/) | Frozen experiment protocol, canonical manifest, split assignments, and integrity reports. |
| [README.md](README.md), [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md), [QUICKSTART.md](QUICKSTART.md), [UNIFIED_SYSTEM_SUMMARY.md](UNIFIED_SYSTEM_SUMMARY.md) | Supporting docs; verify examples against source because some are stale. |
| [CODE_STANDARDS.md](CODE_STANDARDS.md) | Intended coding conventions; existing source/docs are not uniformly aligned. |

Other root scripts (`audio_test.py`, `audio_live_test.py`, `camera_test.py`, `face_test.py`, `haar_face_test.py`, `mic_devices.py`, `mic_test.py`, `yolo_test.py`, `webcam_yolo.py`) are standalone hardware/model utilities. Check each script’s current CLI/source rather than relying on historical quick-start prose.

## 10. Research workflow and dashboard

The repository has a second, separate research track in addition to the root real-time pipeline:

- `research/build_manifest.py` builds the file-level manifest from the extracted dataset and metadata.
- `research/create_split.py` assigns UUID groups to frozen train/validation/test partitions. The checked-in protocol records seed `41041`, stratified 70/15/15 group allocation, 549 UUID groups, and 2,148 WAV files. Do not casually regenerate or manually edit these assignments.
- `research/M2_audio_baseline/extract_features.py` extracts deterministic 256-dimensional log-mel summary features (mono, 16 kHz, 64 mel bins, mean/std/min/max pooling) into `features/`.
- `research/M2_audio_baseline/train_baseline.py` trains a `StandardScaler` + `LogisticRegression` pipeline using training data and evaluates on validation data. `--evaluate-test` loads the frozen model and evaluates the test split only; do not use test results for tuning.
- The remaining M2 scripts validate source audio, inspect validation errors, and analyze metadata. The `results/` and `features/` folders contain generated experiment artifacts, including a saved model and evaluation outputs.
- `research/M3_vision_baseline/` currently contains only a manifest template; do not imply that a vision research baseline is implemented.

`demo/app.py` is a Streamlit mentor/demo dashboard, not a wrapper around `SentinelVisionUnified`:

- Launch from the repository root with `streamlit run demo/app.py` (or `python -m streamlit run demo/app.py`).
- Its live tab captures a still frame and runs YOLO perception. Its event-state selector is explicitly a simulation, not a model result.
- Its audio tab loads the saved M2 scikit-learn model and uses the research log-mel feature extractor to classify the dataset's firearm/caliber labels. It does not run the AST microphone detector.
- Its research tab displays existing metrics, predictions, confusion matrices, and manifest data when those artifacts exist.

The `research/` tree contains scripts and generated features/results; `research_data/` holds dataset archives/extractions and protocol/split artifacts. `benchmarks/`, `benchmark_output/`, `benchmark_outputs/`, `dataset_collected/`, `research_models/`, and `research_results/` are data or run-output locations, not the authoritative runtime implementation. `backup_stabilization_20260902_051149/` and `SentinelVision_Stabilization_Patch/` are historical copies/artifacts.

## 11. Workspace-specific cautions

- Several research scripts and `demo/app.py` hard-code `PROJECT_ROOT = Path(r"D:\SentinalVision")`. The dashboard/research workflow is not currently relocatable without changing that assumption.
- There is no root dependency manifest. A repository-local `venv/` exists, but do not assume it is the selected interpreter or that its installed packages match a contributor's environment.
- Treat source dataset files, `research_data/protocol/master_manifest.csv`, frozen split assignments/configuration, and generated evaluation artifacts as research evidence. Do not overwrite or regenerate them unless the experiment task explicitly calls for it; preserve source data and document any new experiment outputs.
- Dashboard labels such as “Audio Evidence Available” are static UI text, not runtime sensor-health checks. Only report what its underlying action actually computed.

## 3. Runtime data flow

1. `SentinelVisionUnified.process_frame(frame, frame_id)` validates a non-empty color `numpy.ndarray` shaped `(height, width, 3)`.
2. It processes vision synchronously by calling `VisionDetector.process(frame)`.
3. It reads the latest audio result from `AudioDetector`; inference timestamps older than `audio_max_age_sec` (default 3 seconds) are excluded from fusion.
4. It adapts raw results into `AudioFeatures` / `VisionFeatures` and delegates scoring to `FusionEngine`.
5. Fusion computes normalized weighted contributions, applies EMA (unless a configured sudden-evidence reset applies), classifies the smoothed result, and returns availability/contribution/score data.
6. The orchestrator adds reasoning and alert handling and returns a `DetectionResult` with sensor, fusion, health, performance, and annotated-frame data.

The camera is owned by the caller (demo/benchmark/collector), not by `SentinelVisionUnified`. Always release it in a `finally` block and call `system.stop()`.

### Audio behavior

`AudioDetector` opens a mono 16 kHz `sounddevice.InputStream` by default, copies callback samples into a rolling 2-second buffer, and runs AST inference on a daemon thread every 1 second by default. It emits the top predictions and computes a threat score from matching configured threat labels as the capped sum of `confidence × configured class weight`. The default threat-label mapping has 23 entries. Matching depends on exact model label strings; it is not a dedicated gunshot classifier.

Audio result keys include `label`, `confidence`, `threat_score`, `risk_label`, `top_predictions`, `threat_predictions`, `peak`, `rms`, `timestamp`, and `inference_latency_ms`. Empty/uninitialized results use `timestamp=None`; silence can yield a current timestamp with a zero threat score. Audio age in unified telemetry means time since inference timestamp, not the age of the buffered sound samples.

### Vision behavior

`VisionDetector.process(frame)` runs YOLO each frame. Haar face detection runs on the first frame and every third frame by default; cached face count is reused in between. The feature dictionary includes `person_count`, `face_count`, `objects` (name/confidence pairs), `fps`, `timings` (`yolo_ms`, `haar_ms`, `vision_total_ms`), `health_status`, `last_error`, and `device`.

Generic detected objects do not contribute to the current vision threat score. Vision scoring uses only person/crowd count and face-to-person ratio context. There is no identity recognition/tracking or weapon classification.

## 4. Important classes and current configuration

### Fusion policy — configure here

`FusionConfig` is a frozen dataclass and is the authoritative source for fusion behavior. Defaults:

- Weights: audio `0.60`, vision `0.40` (normalized on construction).
- Risk thresholds: suspicious `0.30`, threatening `0.70`; lower values classify as benign.
- EMA alpha `0.30`; sudden-event reset enabled at raw score `0.50` when prior EMA is below suspicious threshold.
- Crowd thresholds: 3 / 5 / 8 people, adding 0.08 / 0.20 / 0.35 respectively.
- Low face/person ratio: below `0.30` adds `0.15` when people are present.
- Vision contextual score is capped at `0.60`.

Use `SentinelUnifiedConfig(fusion=FusionConfig(...))` for new fusion settings. The orchestrator also supports optional constructor `audio_weight` / `vision_weight` overrides for backward compatibility. Do not introduce a second threshold or scoring source.

### Audio configuration

`AudioDetectorConfig` is a plain class with class attributes, not a dataclass. Defaults include model `MIT/ast-finetuned-audioset-10-10-0.4593`, 16 kHz, 2-second buffer, mic device 0, block size 1600, 1-second inference interval, and silence peak threshold 0.01. `AudioDetector(device="auto")` chooses CUDA when available, otherwise CPU; device index can be overridden with `mic_device`.

### Vision configuration

`VisionDetectorConfig` is a plain class with class attributes, not a dataclass. Defaults include model path `yolo26n.pt`, 320 image size, confidence 0.45, Haar width 480, scale factor 1.05, 3 neighbors, min size 40, and Haar interval 3 frames. The constructor’s `yolo_device`, `yolo_imgsz`, and `haar_scale_width` parameters control runtime values; the latter two default explicitly to 320/480, so merely changing those attributes on a config object may not take effect. `YOLO_DEVICE` is not the constructor’s device selector. Verify config flow before changing it.

## 5. Unified output contract

`process_frame()` returns `DetectionResult` (or can return `None` on an internal processing-failure path). Current top-level keys:

- `timestamp`, `frame_id`
- `vision`: `annotated_frame`, `persons`, `faces`, `objects`, `fps`, `health_status`
- `audio`: latest audio result dictionary described above
- `composite`: `risk_label`, `threat_score`, `fusion_confidence`, `sensor_coverage`, `audio_available`, `vision_available`, `audio_contribution`, `vision_contribution`, `raw_audio_score`, `raw_vision_score`, `effective_audio_weight`, `effective_vision_weight`, `reasoning`
- `performance`: `vision_latency_ms`, `audio_age_ms`, `frame_fps`
- `health`: `vision_status`, `audio_status`, `last_vision_error`, `last_audio_error`
- `annotated_frame`: alias of `vision["annotated_frame"]`; it can be `None` if vision is unavailable.

`fusion_confidence` is a backward-compatibility alias for `sensor_coverage`; it is **not** classifier confidence, statistical confidence, or sensor agreement. `audio_age_ms` is observation age, not AST inference latency. Audio inference latency is in the raw `audio` dictionary.

If both sensors are unavailable, the current fusion implementation produces score `0` / label `BENIGN`, while coverage is `0` and both availability flags are false. Callers must gate interpretation on coverage/availability: this output means **no usable sensor evidence**, not positive evidence that the scene is safe.

## 6. Health, concurrency, lifecycle

- Audio callback and inference work occur outside the frame-processing loop. Audio buffer/result access uses a lock; `get_result()` is a shallow copy.
- Vision and fusion run in the caller’s frame thread.
- `get_recent_alerts(count=10)` returns newest-first, and non-positive counts return `[]`. Alerts are queued for suspicious and threatening levels, share a cooldown (default 2 seconds), and are stored in a bounded deque (default 100). `get_status()["alerts_total"]` is current queue length, not lifetime total.
- Graceful degradation is on by default. Detector initialization failures can leave a sensor unavailable while the other continues. Sensor health, availability, freshness, and zero score are distinct concepts: preserve all of them in new consumers.
- `SentinelVisionUnified.stop()` stops audio only. Vision currently has no matching release method; camera release remains the caller’s responsibility.
- Known lifecycle hazard: if audio retrieval fails after initialization, the orchestrator can mark audio unavailable while retaining the detector object, and `stop()` only stops audio while `audio_available` is true. Review this failure path before modifying shutdown or availability behavior.
- Internal vision fallback may produce zero detections while the detector remains initialized; inspect `get_health()` rather than inferring health from a score alone.

## 7. Known issues / discrepancies to keep in view

These were identified by static inspection of current root source/tests; they have not been revalidated by running the hardware-dependent paths.

1. **Benchmark field mismatch:** [benchmark.py](benchmark.py) reads `performance["yolo_ms"]`, `performance["haar_ms"]`, `performance["ast_ms"]`, and `performance["fps"]`. `process_frame()` currently returns only `vision_latency_ms`, `audio_age_ms`, and `frame_fps` in `performance`. A successfully processed frame therefore raises `KeyError` during benchmark metrics extraction. The benchmark display also assumes a non-`None` annotated frame.
2. **Unavailable vision display:** the orchestrator may return `annotated_frame=None`; the demo, benchmark display path, and dataset collector draw/display it without a fallback. Keep headless/degraded cases safe when fixing these consumers.
3. **Camera read retry loops:** demo and dataset collection immediately continue on a failed camera read; benchmark writes a failure row and continues. This can spin without a delay or bounded recovery.
4. **Dataset write/synchronization:** `collect_dataset.py` saves the displayed frame and current rolling audio buffer on a keypress, does not establish exact audio/video alignment, and does not check image-write success.
5. **Config API examples are stale:** docs show calls like `VisionDetectorConfig(YOLO_IMGSZ=...)` and `AudioDetectorConfig(BUFFER_SECONDS=...)`; those plain classes do not accept keyword constructor arguments. The current orchestrator config nests fusion settings under `fusion=FusionConfig(...)`; older docs put weights and thresholds directly into `SentinelUnifiedConfig`.
6. **Demo docs are stale:** documented flags/keys for weights, pause, save, alert display, etc. are not in the current demo. Its current option is `--camera`; `q` quits.
7. **Vision standalone quick-start is stale:** docs say `python vision.py` runs a webcam loop; current `vision.py` is the detector module, not that demo entry point.
8. **Historical summary describes old architecture:** claims of weapon scoring, tracking, cross-modal boosts such as “gunshot + persons”, old helper methods, 40+ tests, or performance/accuracy/uptime results must not be repeated as current verified facts. Current tests affirm that generic weapon labels do not affect vision score.
9. **No dependency manifest:** no root `requirements.txt` or `pyproject.toml` was found. Source imports imply at least NumPy, OpenCV, PyTorch, Ultralytics, Transformers, sounddevice, and soundfile. `yolo26n.pt` is locally present, while the AST model is loaded from Hugging Face. The README’s environment versions are claims, not proof of the active interpreter.
10. **Limited test coverage:** core tests mock detector creation; they do not exercise live hardware, AST callback/inference, actual YOLO/Haar, teardown races, benchmark schema, or dataset writes. `test_audio.py` is empty.

## 8. Tests and development workflow

Tests are `unittest`-based and can be run with pytest or unittest from the repository root:

```powershell
python -m pytest -v
python -m pytest test_fusion_engine.py -v
python -m pytest test_sentinel_unified.py -v
```

Equivalent focused unittest invocation:

```powershell
python -m unittest test_fusion_engine test_sentinel_unified -v
```

The fusion suite covers default weights/thresholds, crowd/face-ratio scoring, ignored weapon/object labels, single-sensor weight renormalization, missing sensors, classification, and EMA reset. The orchestrator suite covers mocked initialization failures, stale audio, weighted results, alert cooldown, output keys, and invalid-frame validation. These tests do not prove live camera/microphone/model behavior.

For hardware-dependent checks, separately confirm camera/microphone availability, model assets/download access, and CPU/GPU selection. The README reports a Windows/Python 3.11.9 CPU-PyTorch environment, but agents should inspect the selected environment before relying on that. Do not run the long benchmark or live demo as a routine unit test.

## 9. Agent contribution guidance

1. Read the relevant root source and tests first; use historical documentation only as background.
2. For scoring changes, keep implementation in `FusionConfig` / `FusionEngine`; add deterministic unit tests for thresholds, availability, missing inputs, and temporal state.
3. For schema changes, update the `DetectionResult` type and every consumer: demo, quickstart, benchmark, collector, tests, and current docs.
4. When touching sensors, preserve distinctions between **availability**, **health**, **freshness**, **classifier confidence**, **threat score**, and **coverage**. Never turn absent/stale evidence into a claim of safety.
5. For fixes, write/adjust a focused test first where possible, run the focused suite, then the full suite. Keep camera, microphone, model download, and GPU validation separate.
6. Ensure camera/audio resources are cleaned up on every exit path. Handle `None` annotated frames and sensor initialization failures explicitly.
7. Follow [CODE_STANDARDS.md](CODE_STANDARDS.md) where practical: type hints, Google-style docstrings, specific errors, logging, centralized configuration, and `try/finally` resource cleanup. Do not mistake its examples or older docs for verified current APIs.
8. Do not claim production readiness, calibrated threat probability, weapon detection, accuracy gains, or performance numbers without a reproducible experiment and documented hardware/data/protocol.
