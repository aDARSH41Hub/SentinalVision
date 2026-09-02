# SentinelVision

**SentinelVision** is a real-time multimodal surveillance research prototype that combines **visual** and **audio** evidence to assess potentially threatening events.

The system currently combines:

* **YOLO-based visual perception** for people and general scene objects
* **Haar Cascade** face detection
* **Audio Spectrogram Transformer (AST)** for acoustic-event classification
* **Rule-based multimodal evidence fusion**
* **Exponential Moving Average (EMA)** temporal smoothing
* **Sensor availability and health monitoring**
* **Benchmarking and controlled dataset collection utilities**

The primary research objective is to investigate whether combining complementary audio-visual evidence can produce more robust event assessment than relying on either modality independently.

> **Research status:** Phase 1 baseline is implemented. Cross-modal attention and LSTM-based temporal modeling are planned research stages and are **not yet part of the current baseline**.

---

## 1. Research Objective

The central research hypothesis is:

> A multimodal surveillance system that combines complementary audio and visual evidence can improve threat-event discrimination compared with independent single-modality systems while maintaining practical real-time computational performance.

The project is intended to answer three primary research questions:

### RQ1 — Multimodal Fusion

Does combining audio and visual evidence improve event/threat classification compared with audio-only and vision-only baselines?

### RQ2 — Temporal Robustness

Does temporal smoothing reduce the effect of isolated noisy predictions and short-lived false alarms?

### RQ3 — Real-Time Feasibility

Can the complete pipeline operate with practical latency and throughput on commodity hardware?

A later research stage will investigate:

### RQ4 — Learned Fusion

Can learned cross-modal attention outperform fixed-weight rule-based fusion?

---

# 2. Current System Status

| Component                           | Status            |
| ----------------------------------- | ----------------- |
| YOLO visual detection               | ✅ Implemented     |
| Face detection                      | ✅ Implemented     |
| AST audio classification            | ✅ Implemented     |
| Audio timestamps                    | ✅ Implemented     |
| Audio freshness handling            | ✅ Implemented     |
| Centralized fusion engine           | ✅ Implemented     |
| Sensor availability handling        | ✅ Implemented     |
| Sensor health telemetry             | ✅ Implemented     |
| EMA temporal smoothing              | ✅ Implemented     |
| Runtime demo                        | ✅ Implemented     |
| Benchmarking                        | ✅ Implemented     |
| Controlled dataset collection       | ✅ Implemented     |
| Dedicated weapon detector           | ⏳ Future research |
| Gunshot-specific trained classifier | ⏳ Planned         |
| Cross-modal attention               | ⏳ Planned         |
| 16-frame LSTM temporal model        | ⏳ Planned         |
| Final research evaluation           | ⏳ Pending         |

---

# 3. Architecture

```text
                    ┌──────────────────────┐
                    │    Video Stream      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Vision Module     │
                    │                      │
                    │ YOLO + Haar Cascade  │
                    └──────────┬───────────┘
                               │
                    Vision Features
                               │
                               ▼
                      ┌─────────────────┐
                      │                 │
                      │  Fusion Engine  │
                      │                 │
                      │ Audio Evidence  │
                      │       +         │
                      │ Vision Evidence │
                      │       │         │
                      │       ▼         │
                      │  EMA Smoothing  │
                      │                 │
                      └────────┬────────┘
                               │
                         Fused Score
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Threat / Risk Level  │
                    │                      │
                    │ BENIGN               │
                    │ SUSPICIOUS           │
                    │ THREATENING          │
                    └──────────────────────┘
                               ▲
                               │
                    Audio Features
                               │
                    ┌──────────┴───────────┐
                    │    Audio Module      │
                    │                      │
                    │ AST Audio Classifier │
                    └──────────┬───────────┘
                               │
                    ┌──────────────────────┐
                    │     Microphone       │
                    └──────────────────────┘
```

---

# 4. Core Components

## 4.1 Visual Perception

The visual pipeline uses a YOLO-based object detector together with Haar Cascade face detection.

The current visual module extracts primarily:

* Person count
* Face count
* General detected object names
* Processing latency
* Detector health state

The current system does **not** claim that a generic pretrained YOLO model is a scientifically validated weapon detector.

Visual threat scoring is currently contextual and heuristic.

For example, contextual evidence can increase when:

* a large number of people are detected
* the detected person count is inconsistent with detected faces

This should be understood as **scene/context evidence**, not semantic weapon recognition.

---

# 5. Audio Perception

The audio pipeline uses an **Audio Spectrogram Transformer (AST)** model for acoustic-event classification.

The module provides:

* Dominant predicted audio class
* Classifier confidence
* Audio-derived threat score
* Top-k predictions
* Timestamp
* Inference latency
* Sensor health state

The pipeline maintains a rolling audio buffer and periodically performs inference.

The audio output is treated as evidence rather than as an absolute statement that a threat is present.

For future research, the audio pipeline will be evaluated specifically for **gunshot vs. non-gunshot acoustic-event discrimination**, followed by broader acoustic-event evaluation.

---

# 6. Multimodal Fusion

The current fusion stage uses a deterministic weighted evidence model.

Let:

* \(A_t\) = audio threat score
* \(V_t\) = vision threat score
* \(w_a\) = audio weight
* \(w_v\) = vision weight

The raw multimodal score is:

$$
S_t^{raw} =
w_a A_t +
w_v V_t
$$

The current baseline uses:

$$
w_a = 0.60
$$

$$
w_v = 0.40
$$

The fusion engine renormalizes the weights when one sensor becomes unavailable.

This is important because **missing sensor evidence is not equivalent to zero threat evidence**.

For example:

```text
Audio unavailable
        ↓
Do not treat audio as "BENIGN"
        ↓
Use available vision evidence
        ↓
Renormalize active sensor weights
```

---

# 7. Temporal Smoothing

The baseline uses an Exponential Moving Average (EMA):

$$
S_t =
\alpha S_t^{raw}
+
(1-\alpha)S_{t-1}
$$

where:

* \(S_t^{raw}\) is the current fused score
* \(S_{t-1}\) is the previous smoothed score
* \(\alpha\) controls responsiveness

The current baseline configuration uses:

$$
\alpha = 0.30
$$

EMA is intended to reduce sensitivity to isolated noisy predictions.

A reset mechanism is also available for strong sudden evidence so that temporal smoothing does not unnecessarily delay an abrupt high-confidence event.

---

# 8. Threat Levels

The baseline maps the fused score to three levels:

| Level       | Score Range     |
| ----------- | --------------- |
| BENIGN      | `< 0.30`        |
| SUSPICIOUS  | `0.30 – < 0.70` |
| THREATENING | `>= 0.70`       |

These thresholds are configurable and should ultimately be validated using experimental data rather than treated as universally optimal.

---

# 9. Sensor Health and Missing Data

A major design principle is that **sensor failure must not silently become benign evidence**.

The unified pipeline tracks:

```text
vision_status
audio_status
last_vision_error
last_audio_error
```

Possible sensor states include:

```text
OK
DEGRADED
ERROR
```

The system also tracks whether each modality is currently available for fusion.

This allows the runtime system to distinguish:

```text
"No evidence of threat"
```

from:

```text
"Sensor failed, therefore no evidence was available"
```

That distinction is important for both engineering reliability and scientific evaluation.

---

# 10. Project Structure

Current project structure:

```text
SentinalVision/
│
├── audio.py
├── audio_test.py
├── audio_live_test.py
│
├── vision.py
├── camera_test.py
├── face_test.py
├── haar_face_test.py
├── webcam_yolo.py
├── yolo_test.py
│
├── fusion_engine.py
│
├── sentinel_vision.py
├── demo_sentinel_unified.py
├── quickstart_unified.py
│
├── benchmark.py
├── collect_dataset.py
│
├── test_audio.py
├── test_fusion_engine.py
├── test_sentinel_unified.py
│
├── CODE_STANDARDS.md
├── ENHANCEMENTS.md
├── ENHANCEMENT_REPORT.md
├── INTEGRATION_GUIDE.md
├── QUICKSTART.md
├── UNIFIED_SYSTEM_SUMMARY.md
│
└── README.md
```

Experimental backups and the local Python virtual environment are intentionally excluded from version control.

---

# 11. Environment

The current development environment is:

```text
OS: Windows
Python: 3.11.9
OpenCV: 4.10.0
PyTorch: 2.13.0+cpu
Transformers: 5.15.1
```

The project currently runs with a CPU PyTorch build.

Although the development machine has an NVIDIA RTX 3050, CUDA acceleration is not currently enabled in the installed PyTorch environment.

This means reported performance should **not** be interpreted as GPU-accelerated performance unless a CUDA-enabled environment is explicitly used for the experiment.

---

# 12. Installation

Open PowerShell:

```powershell
cd D:\SentinalVision
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

Expected:

```text
Python 3.11.9
```

Install dependencies from the project's dependency configuration when available.

---

# 13. Running the System

## Unified Demo

```powershell
python demo_sentinel_unified.py
```

The runtime HUD reports information such as:

```text
Threat score
Threat level
Person count
Face count
Audio class
Audio threat score
Vision score
Sensor coverage
FPS
Vision latency
Audio age
```

Sensor errors are surfaced explicitly rather than being silently converted to a benign state.

---

# 14. Running Tests

Run the fusion-engine tests:

```powershell
python -m pytest test_fusion_engine.py -v
```

Run unified-system tests:

```powershell
python -m pytest test_sentinel_unified.py -v
```

Run the complete test suite:

```powershell
python -m pytest -v
```

---

# 15. Benchmarking

The project includes `benchmark.py` for reproducible runtime measurements.

The benchmark records per-frame information such as:

```text
timestamp
frame_id
YOLO latency
Haar latency
AST latency
fused score
risk level
person count
face count
audio class
audio threat score
FPS
```

The benchmark also calculates summary statistics such as:

* Mean latency
* Median latency
* P95 latency
* Minimum FPS
* Maximum FPS
* Frame drops

Example:

```powershell
python benchmark.py
```

For headless execution:

```powershell
python benchmark.py --no-window
```

The default benchmark duration is intended for practical runtime measurement and can be configured from the command line.

---

# 16. Dataset Collection

`collect_dataset.py` provides a controlled data-collection workflow.

Keyboard labels:

```text
B = Benign
S = Suspicious
T = Threatening
```

A capture stores:

```text
Image frame
Rolling audio segment
Metadata JSON
```

The dataset collector is intended for:

* controlled
* staged
* acted

scenarios.

Do not use real weapons or unsafe situations for data collection.

The collected dataset should be treated as experimental research data and documented with:

* scenario definition
* recording environment
* labels
* sensor configuration
* sampling rate
* timestamp information
* train/validation/test protocol

---

# 17. Experimental Plan

The research will progress through controlled baselines.

## Experiment 1 — Vision Only

Evaluate the visual pipeline independently.

Measure:

* precision
* recall
* mAP@0.5
* mAP@0.5:0.95
* latency
* FPS

Object-detection metrics should not be treated as direct measurements of overall threat-classification quality.

---

## Experiment 2 — Audio Only

Evaluate the audio pipeline independently.

The first focused audio task is:

```text
Gunshot vs. non-gunshot
```

Later experiments can investigate additional acoustic-event classes.

Measure:

* accuracy
* precision
* recall
* F1
* confusion matrix
* inference latency

---

## Experiment 3 — Rule-Based Multimodal Fusion

Evaluate:

```text
Audio only
Vision only
Audio + Vision
```

This establishes the current multimodal baseline.

---

## Experiment 4 — Ablation Study

Possible comparisons:

```text
Audio only
Vision only
Equal-weight fusion
Configured-weight fusion
Fusion without EMA
Fusion with EMA
```

The objective is to identify which components actually contribute to performance.

---

## Experiment 5 — Cross-Modal Attention

A later learned-fusion model will investigate whether learned interactions between audio and visual representations outperform the fixed-weight baseline.

---

## Experiment 6 — Temporal LSTM

A later stage will evaluate temporal sequences, targeting the proposed 16-frame temporal modeling stage.

The comparison should be against the same baseline and evaluation protocol rather than against unrelated systems.

---

# 18. Scientific Positioning

The current implementation should be described as:

> A lightweight multimodal audio-visual evidence fusion baseline for real-time surveillance event assessment.

It should **not** currently be described as:

* a fully autonomous security system
* a scientifically validated weapon detector
* a production-grade surveillance platform
* a proven gunshot detection system
* a learned cross-modal attention architecture
* an LSTM-based temporal model

Those are future or experimental directions unless supported by measured results.

---

# 19. Current Limitations

Several limitations are intentionally documented.

### Generic visual detection

The current YOLO component is primarily a general object detector and contextual visual feature extractor.

Dedicated weapon detection remains a future research stage requiring an appropriate dataset and evaluation protocol.

### Heuristic fusion

The current fusion mechanism uses fixed weights and manually selected thresholds.

This provides interpretability and simplicity but does not automatically learn optimal cross-modal relationships.

### Audio-domain generalization

Audio classification can be sensitive to:

* microphone characteristics
* distance
* background noise
* reverberation
* recording environment
* class imbalance

External datasets and controlled recordings are therefore required for meaningful evaluation.

### Temporal modeling

EMA provides lightweight temporal stabilization but is not equivalent to learned sequence modeling.

The proposed LSTM stage is intended to investigate richer temporal dependencies.

### Hardware configuration

Current development uses a CPU PyTorch environment, so runtime measurements must clearly report the execution hardware and software configuration.

---

# 20. Research Roadmap

```text
Phase 1
│
├── Vision pipeline
├── Audio pipeline
├── Centralized fusion
├── Sensor health
└── EMA stabilization
        │
        ▼
Phase 2
│
├── Reproducible benchmark
├── Audio dataset evaluation
├── Vision baseline evaluation
├── Multimodal baseline evaluation
└── Ablation studies
        │
        ▼
Phase 3
│
├── Cross-modal attention
└── Learned fusion evaluation
        │
        ▼
Phase 4
│
├── Temporal LSTM
├── Sequence evaluation
└── Temporal ablation
        │
        ▼
Phase 5
│
├── Final experiments
├── Error analysis
├── Statistical comparison
└── IEEE research paper
```

---

# 21. Reproducibility

Every reported research result should document:

```text
Model version
Dataset version
Dataset split
Random seed
Hardware
Python version
Major package versions
Configuration
Thresholds
Fusion weights
Evaluation metrics
```

No result should be added to the research paper unless it can be reproduced from the recorded experiment configuration.

---

# 22. Repository and Versioning

Git is used to maintain the evolution of the research system.

Recommended commit progression:

```text
phase1: establish vision pipeline
phase1: establish audio pipeline
phase1: add audio timestamps
phase1: implement centralized fusion engine
phase1: add sensor availability handling
phase1: add temporal smoothing
phase1: stabilize unified pipeline
research: add benchmark instrumentation
research: add dataset collection protocol
research: establish audio baseline
research: establish vision baseline
research: establish multimodal baseline
research: add cross-modal attention
research: add temporal LSTM
```

Suggested future tags:

```text
v0.1-phase1-baseline
v0.2-rule-fusion
v0.3-attention
v1.0-final-experiment
```

---

# 23. Research Paper

The working research-paper title is:

**SentinelVision: A Multimodal Audio-Visual Fusion Framework for Real-Time Surveillance**

Alternative:

**SentinelVision: Lightweight Audio-Visual Evidence Fusion for Real-Time Surveillance Event Assessment**

Proposed paper structure:

```text
I.   Introduction
II.  Related Work
III. Research Gap and Problem Formulation
IV.  Proposed SentinelVision Framework
     A. Visual Perception
     B. Audio Perception
     C. Multimodal Fusion
     D. Temporal Smoothing
V.   Experimental Methodology
VI.  Results and Ablation
VII. Error Analysis
VIII. Limitations
IX.  Conclusion
     References
```

The paper will distinguish clearly between:

```text
Implemented
Measured
Planned
```

No performance value should be presented as achieved unless it has actually been measured.

---

# 24. Status

**Current milestone: Phase 1 baseline stabilized.**

The immediate research priority is no longer adding random functionality.

The next goal is to establish a **reproducible experimental baseline** so that later attention and LSTM models can be evaluated against measurable evidence.

---

## License

License information will be added once the project's intended distribution model is finalized.
