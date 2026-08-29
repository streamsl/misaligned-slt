from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json

# Minimum measured (Δ_head, Δ_tail) pairs before raw replay beats the Laplace fitted to the same measurement:
# below this, replay censors the tails (cannot exceed observed extremes) and repeats each value many times per
# epoch, while the 2-parameter fit is already well-estimated. Depends on sample count, not corpus.
RAW_REPLAY_MIN_PAIRS = 1000


@dataclass
class JitterSampler:
    samples: np.ndarray | None = None
    head_loc_s: float = 0.0
    head_scale_s: float = 0.5
    tail_loc_s: float = 0.0
    tail_scale_s: float = 0.5
    # Empirical relative cut positions of over-segmentation events (segmenter-error analysis, analyze.py).
    # Mode 2 draws its truncation depth from these; empty → uniform interior fallback.
    cut_positions: np.ndarray | None = None
    cut_lo: float = 0.15
    cut_hi: float = 0.85

    @classmethod
    def from_config(cls, cfg: dict) -> "JitterSampler":
        cut_range = [float(x) for x in cfg.get("fallback_cut_range", [0.15, 0.85])]
        if len(cut_range) != 2 or not 0.0 < cut_range[0] < cut_range[1] < 1.0: raise ValueError(
            f"jitter.fallback_cut_range must be [lo, hi] inside (0,1); got {cut_range}"
        )
        cut_positions = None
        source = cfg.get("source")
        if source and not Path(source).exists():
            # Like the sampler's mode_ratios guard: configured-but-missing must FAIL, not silently train
            # fallback_laplace while claiming measured jitter. `source: null` = designed, explicitly.
            raise FileNotFoundError(
                f"jitter.source is set but missing: {source!r} (cwd={Path.cwd()}). Run segmenter-error analysis first, or set "
                f"jitter.source: null to explicitly use fallback_laplace."
            )
        if source and Path(source).exists():
            data = json.loads(Path(source).read_text(encoding="utf-8"))
            cuts = np.asarray(data.get("overseg_cut_positions", []), dtype=np.float32).reshape(-1)
            if cuts.size: cut_positions = cuts
            
            raw = data.get("samples", data.get("jitter_samples", []))
            samples = np.asarray([[
                x.get("delta_head_s", x.get("head_s")), x.get("delta_tail_s", x.get("tail_s"))]
                if isinstance(x, dict) else x for x in raw
            ], dtype=np.float32)

            # Large measurement → raw replay; small → the Laplace fitted to the same offsets in the same file
            # (RAW_REPLAY_MIN_PAIRS). Both are measured calibration, never the designed fallback.
            n_pairs = samples.reshape(-1, 2).shape[0] if samples.size else 0
            if n_pairs >= RAW_REPLAY_MIN_PAIRS:
                print(f"[jitter] {source}: RAW empirical replay ({n_pairs} measured pairs)", flush=True)
                return cls(samples=samples.reshape(-1, 2), cut_positions=cut_positions, cut_lo=cut_range[0], cut_hi=cut_range[1])
            print(f"[jitter] {source}: fitted-Laplace draws ({n_pairs} measured pairs < {RAW_REPLAY_MIN_PAIRS}; "
                  f"raw replay would censor tails and lattice the distribution)", flush=True)
            laplace = data.get("laplace", {})
            # Same fail-loud rule as the missing-file guard above: without a fit we would silently draw the
            # hard-coded 0.0/0.5 defaults while claiming measured jitter (the print above would be a lie).
            if not laplace: raise ValueError(
                f"jitter.source {source!r} has {n_pairs} pairs (< {RAW_REPLAY_MIN_PAIRS}) and no 'laplace' fit to fall back on. "
                f"Re-run segmenter-error analysis (it writes both), or set jitter.source: null for the designed mix."
            )
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
            cut_positions=cut_positions, cut_lo=cut_range[0], cut_hi=cut_range[1],
        )

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.samples is not None and len(self.samples):
            idx = int(rng.integers(0, len(self.samples)))
            return float(self.samples[idx, 0]), float(self.samples[idx, 1])
        return (
            float(rng.laplace(self.head_loc_s, self.head_scale_s)),
            float(rng.laplace(self.tail_loc_s, self.tail_scale_s)),
        )

    def sample_cut(self, rng: np.random.Generator) -> float:
        """Relative position in (0,1) of a Mode-2 spurious internal cut.

        From segmenter-errors when available. Otherwise draw uniformly from `fallback_cut_range`; the interior
        range avoids nearly empty windows and is independent of matched-pair boundary jitter.
        """
        if self.cut_positions is not None and len(self.cut_positions):
            return float(self.cut_positions[int(rng.integers(0, len(self.cut_positions)))])
        return float(rng.uniform(self.cut_lo, self.cut_hi))


def normalized_mode_ratios(raw: dict[str, float]) -> dict[str, float]:
    values = {k: max(float(v), 0.0) for k, v in raw.items()}
    total = sum(values.values())
    if total <= 0: raise ValueError("mode ratios need positive mass; set mode_ratios.fallback in the training config")
    return {k: v / total for k, v in values.items()}
