"""Low-rank adapters for the shared mT5 stack (Hu et al. 2021), injected in place.

This module implements the LoRA update W x + (alpha/r) B A x with the base weights frozen.
It retains the project's parameter paths and uses PyTorch operations. The factors use the initialization in
Microsoft's loralib: Kaiming-uniform A and zero B (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py).

`lora_B` is zero-initialised, so an adapted model starts as an exact copy of the base function, and `dL/dA` is exactly 0 on 
1st backward. That is LoRA's design, not a dead path: once `B` moves, `A` receives gradient.

AR and DLM are separate training runs with the same adapter settings. Within a DLM run the manual OPUT forward
and the native cached forward use the same decoder modules and adapter factors.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Linear):
    """An `nn.Linear` that adds `B @ A @ x * alpha/r` to its own output.

    Adopted by reassigning `__class__` on the existing module: the base `weight` and `bias` Parameters keep their
    identity and their names, so an optimizer, a strict load or a state-dict re-key sees exactly what it saw before.
    """
    def adopt(self, rank: int, alpha: float, dropout: float) -> None:
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=self.weight.device, dtype=self.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=self.weight.device, dtype=self.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_scaling = float(alpha) / float(rank)
        self.lora_dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_dropout.train(self.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = nn.functional.linear(nn.functional.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return super().forward(x) + delta * self.lora_scaling


def apply_lora(root: nn.Module, target_modules=("q", "v"), rank: int = 16, alpha: float = 32.0, dropout: float = 0.0) -> dict:
    """Adapt every `nn.Linear` under `root` whose attribute name is in `target_modules`, and freeze the rest of it.

    Call AFTER the pretrained weights are loaded: injection leaves the tree strict-loadable, but a checkpoint saved before it 
    carries no `lora_A` / `lora_B` keys. `eval.py` rebuilds the same adapter from the stamped meta before its own strict load, 
    which is what proves an adapted checkpoint matches the model that produced it.

    T5 names its attention projections `q`, `k`, `v`, `o` in self- and cross-attention alike, so `("q", "v")` covers the mT5 
    encoder and decoder self-attention and the decoder cross-attention. A backbone that names them differently (mBART uses 
    `q_proj` / `v_proj`) matches nothing and raises, rather than silently training nothing.

    Returns the meta the trainer stamps in the checkpoint and `eval.py` reads back.
    """
    # A YAML scalar (`target_modules: q_proj`) would tuple() into ('q','_','p','r','o','j'), silently adapt mT5's 
    # `q` alone and stamp the exploded list in the checkpoint. The guard below only fires when NOTHING matches.
    if isinstance(target_modules, str): raise ValueError(
        f"LoRA target_modules must be a list, got the string {target_modules!r}; write [{target_modules}] in the config."
    )
    if int(rank) <= 0: raise ValueError(f"LoRA rank must be positive, got {rank}")
    # alpha 0 scales every update to 0: the adapter would train and change nothing, which reads as a dead run.
    if float(alpha) == 0.0: raise ValueError("LoRA alpha must be non-zero; alpha/rank scales the whole update")
    if not math.isfinite(float(alpha)) or float(alpha) < 0: raise ValueError("LoRA alpha must be finite and positive")
    if not math.isfinite(float(dropout)) or not 0 <= float(dropout) < 1: raise ValueError("LoRA dropout must be in [0, 1)")
    
    targets = tuple(target_modules)
    candidates = [(path, module) for path, module in root.named_modules()
                  if path.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear)]
    missing = set(targets) - {path.rsplit(".", 1)[-1] for path, _ in candidates}
    if not candidates or missing:
        raise ValueError(f"No nn.Linear named {sorted(missing or set(targets))} under {type(root).__name__}")
    if any(isinstance(module, LoRALinear) for _, module in candidates): raise ValueError("Model already carries LoRALinear")
    adapted = []
    for path, module in candidates:
        module.__class__ = LoRALinear
        module.adopt(int(rank), float(alpha), float(dropout))
        adapted.append(path)

    for name, param in root.named_parameters(): param.requires_grad_(name.endswith(".lora_A") or name.endswith(".lora_B"))
    return {
        "rank": int(rank), "alpha": float(alpha), "dropout": float(dropout), 
        "target_modules": list(targets), "adapted_modules": len(adapted),
        "trainable_params": int(sum(p.numel() for p in root.parameters() if p.requires_grad)),
        "frozen_params": int(sum(p.numel() for p in root.parameters() if not p.requires_grad)),
    }
