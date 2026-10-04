from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


def classify_probability(probability: float, threshold: float = 0.49) -> str:
    """Convert a fused score into a user-facing decision label.

    The research runtime uses a threshold of 0.49 for the alert boundary, but the
    product-facing label bands can still be kept stable and readable:

      - normal: below alert threshold and below the suspicious band
      - suspicious: moderate risk band
      - threat: clearly dangerous band
    """
    score = float(np.clip(probability, 0.0, 1.0))

    if score < max(0.5, threshold):
        return "normal"

    if score < 0.8:
        return "suspicious"

    return "threat"


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    config_path: Path
    model_dir: Path
    audio_model_path: Path
    rgb_model_path: Path
    vggish_root: Path
    rgb_root: Path
    feature_module_path: Path
    target_timesteps: int
    audio_dim: int
    rgb_dim: int
    threshold: float
    audio_alpha: float
    rgb_alpha: float
    device: str

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "RuntimeConfig":
        root = Path(project_root)
        model_dir = root / "research" / "M3_5_multimodal" / "results" / "temporal_late_fusion"
        config_path = model_dir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Missing runtime config: {config_path}")

        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        vggish_root = root / "research_data" / "multimodal" / "xd_violence" / "features" / "vggish"
        rgb_root = root / "research_data" / "multimodal" / "xd_violence" / "features" / "RGB" / "RGB"
        feature_module_path = root / "research" / "M3_5_multimodal" / "m8_realtime_benchmark.py"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            project_root=root,
            config_path=config_path,
            model_dir=model_dir,
            audio_model_path=model_dir / "temporal_audio_best.pt",
            rgb_model_path=model_dir / "temporal_rgb_best.pt",
            vggish_root=vggish_root,
            rgb_root=rgb_root,
            feature_module_path=feature_module_path,
            target_timesteps=int(config.get("target_timesteps", 256)),
            audio_dim=int(config.get("audio_dim", 128)),
            rgb_dim=int(config.get("rgb_dim", 1024)),
            threshold=float(config.get("selected_threshold", 0.49)),
            audio_alpha=float(config.get("selected_alpha_audio", 0.51)),
            rgb_alpha=float(config.get("selected_alpha_rgb", 0.49)),
            device=device,
        )


class RuntimeInferenceService:
    """Runtime entry point for the frozen TLF model selection used in M3.5."""

    def __init__(self, project_root: str | Path = ".") -> None:
        self.config = RuntimeConfig.from_project_root(project_root)
        self.device = torch.device(self.config.device)
        self.benchmark_module = None
        self.audio_index = None
        self.rgb_index = None
        self.audio_model = None
        self.rgb_model = None
        self._loaded = False

    def load_runtime_models(self) -> None:
        if self._loaded:
            return

        self.benchmark_module = self._load_feature_module(self.config.feature_module_path)
        self.audio_index, self.rgb_index = self._build_feature_index()
        self.audio_model = self._load_model(self.config.audio_model_path, self.config.audio_dim)
        self.rgb_model = self._load_model(self.config.rgb_model_path, self.config.rgb_dim)
        self.audio_model.to(self.device)
        self.rgb_model.to(self.device)
        self.audio_model.eval()
        self.rgb_model.eval()
        self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_runtime_models()

    @staticmethod
    def _load_feature_module(module_path: Path):
        if not module_path.exists():
            raise FileNotFoundError(f"Missing benchmark module: {module_path}")

        spec = importlib.util.spec_from_file_location("m8_runtime_module", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import module {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _build_feature_index(self) -> Tuple[Dict[str, Path], Dict[str, Dict[int, Path]]]:
        vggish_files = sorted(self.config.vggish_root.rglob("*.npy"))
        rgb_files = sorted(self.config.rgb_root.rglob("*.npy"))

        audio_index: Dict[str, Path] = {}
        for path in vggish_files:
            key = path.stem
            if key in audio_index:
                raise RuntimeError(f"Duplicate audio feature key: {key}")
            audio_index[key] = path

        rgb_index: Dict[str, Dict[int, Path]] = {}
        for path in rgb_files:
            stem = path.stem
            crop = None
            base = None
            for crop_id in range(5):
                suffix = f"__{crop_id}"
                if stem.endswith(suffix):
                    crop = crop_id
                    base = stem[: -len(suffix)]
                    break
            if crop is None:
                continue
            rgb_index.setdefault(base, {})[crop] = path

        if not audio_index:
            raise FileNotFoundError(f"No VGGish features found under {self.config.vggish_root}")
        if not rgb_index:
            raise FileNotFoundError(f"No RGB features found under {self.config.rgb_root}")

        return audio_index, rgb_index

    def _load_model(self, checkpoint_path: Path, input_dim: int):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_cls = self.benchmark_module.TemporalGRU
        model = model_cls(
            input_dim=input_dim,
            projection_dim=int(checkpoint.get("projection_dim", 128)),
            hidden_dim=int(checkpoint.get("hidden_dim", 128)),
            num_layers=int(checkpoint.get("num_layers", 1)),
            dropout=0.0,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model

    def predict_segment(self, segment_id: str) -> Dict[str, float | str]:
        self._ensure_loaded()

        audio_feature = self.benchmark_module.load_audio(segment_id, self.audio_index)
        rgb_feature = self.benchmark_module.load_rgb(segment_id, self.rgb_index)

        audio_tensor = torch.from_numpy(audio_feature).to(self.device).unsqueeze(0).contiguous()
        rgb_tensor = torch.from_numpy(rgb_feature).to(self.device).unsqueeze(0).contiguous()

        with torch.inference_mode():
            audio_logits = self.audio_model(audio_tensor)
            rgb_logits = self.rgb_model(rgb_tensor)
            audio_probability = float(torch.sigmoid(audio_logits).squeeze(0).cpu().item())
            rgb_probability = float(torch.sigmoid(rgb_logits).squeeze(0).cpu().item())

        fused_probability = (
            self.config.audio_alpha * audio_probability + self.config.rgb_alpha * rgb_probability
        )
        decision_label = classify_probability(fused_probability, threshold=self.config.threshold)

        return self.build_decision_payload(
            audio_probability=audio_probability,
            rgb_probability=rgb_probability,
            fused_probability=fused_probability,
            decision_label=decision_label,
            decision_score=fused_probability,
            model_name="TLF",
        )

    @staticmethod
    def build_decision_payload(
        audio_probability: float,
        rgb_probability: float,
        fused_probability: float,
        decision_label: str,
        decision_score: float,
        model_name: str,
        threshold: float = 0.49,
    ) -> Dict[str, float | str]:
        return {
            "model_name": model_name,
            "audio_probability": float(audio_probability),
            "rgb_probability": float(rgb_probability),
            "fused_probability": float(fused_probability),
            "decision_label": decision_label,
            "decision_score": float(decision_score),
            "threshold": float(threshold),
            "alert_triggered": float(fused_probability) >= float(threshold),
        }


if __name__ == "__main__":
    service = RuntimeInferenceService(project_root="D:/SentinalVision")
    segment_id = "Bad.Boys.1995__#00-26-51_00-27-53_label_B2-0-0__vggish"
    result = service.predict_segment(segment_id)
    print(result)
