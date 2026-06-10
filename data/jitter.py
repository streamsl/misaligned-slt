from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json


@dataclass
class JitterSampler:
    samples: np.ndarray | None = None
    head_loc_s: float = 0.0
    head_scale_s: float = 0.5
    tail_loc_s: float = 0.0
    tail_scale_s: float = 0.5
    # Empirical relative cut positions of over-segmentation events (Analysis A). Mode 2 draws its
    # truncation depth from these; empty → uniform interior fallback. See analyze.py.
    cut_positions: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> "JitterSampler":
        cut_positions = None
        source = cfg.get("source")
        if source and Path(source).exists():
            data = json.loads(Path(source).read_text(encoding="utf-8"))
            cuts = np.asarray(data.get("overseg_cut_positions", []), dtype=np.float32).reshape(-1)
            if cuts.size: cut_positions = cuts
            
            raw = data.get("samples", data.get("jitter_samples", []))
            samples = np.asarray([[
                x.get("delta_head_s", x.get("head_s")), x.get("delta_tail_s", x.get("tail_s"))]
                if isinstance(x, dict) else x for x in raw
            ], dtype=np.float32)

            if samples.size: return cls(samples=samples.reshape(-1, 2), cut_positions=cut_positions)
            laplace = data.get("laplace", {})
        else: laplace = cfg.get("fallback_laplace", {})

        if "head" in laplace or "tail" in laplace:
            head = laplace.get("head", {})
            tail = laplace.get("tail", {})
            laplace = {
                "head_loc_s": head.get("loc", 0.0),
                "head_scale_s": head.get("scale", 0.5),
                "tail_loc_s": tail.get("loc", 0.0),
                "tail_scale_s": tail.get("scale", 0.5),
            }
        return cls(
            head_loc_s=float(laplace.get("head_loc_s", 0.0)),
            head_scale_s=max(float(laplace.get("head_scale_s", 0.5)), 1e-6),
            tail_loc_s=float(laplace.get("tail_loc_s", 0.0)),
            tail_scale_s=max(float(laplace.get("tail_scale_s", 0.5)), 1e-6),
            cut_positions=cut_positions,
        )

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.samples is not None and len(self.samples):
            idx = int(rng.integers(0, len(self.samples)))
            return float(self.samples[idx, 0]), float(self.samples[idx, 1])
        return (
            float(rng.laplace(self.head_loc_s, self.head_scale_s)),
            float(rng.laplace(self.tail_loc_s, self.tail_scale_s)),
        )

    def sample_cut(self, rng: np.random.Generator, lo: float = 0.15, hi: float = 0.85) -> float:
        """Relative position in (0,1) of a Mode-2 spurious internal cut.

        Drawn from Analysis A's measured over-segmentation cut positions when available; otherwise a
        uniform interior cut in [lo, hi] (avoids degenerate near-empty windows). Uniform is the
        noninformative prior — over-seg internal-cut positions are not in the matched-pair jitter CDF.
        """
        if self.cut_positions is not None and len(self.cut_positions):
            return float(self.cut_positions[int(rng.integers(0, len(self.cut_positions)))])
        return float(rng.uniform(lo, hi))


def normalized_mode_ratios(raw: dict[str, float]) -> dict[str, float]:
    values = {k: max(float(v), 0.0) for k, v in raw.items()}
    total = sum(values.values())
    if total <= 0: return {"mode1": 0.55, "mode2": 0.20, "mode3": 0.20, "mode4": 0.05}
    return {k: v / total for k, v in values.items()}
