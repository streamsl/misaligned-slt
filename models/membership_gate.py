"""The membership gate Ω(t): conditioning the translation decoder on the BIO head's segmentation beliefs.

Implements `docs/membership_gate.md` §2.9 verbatim — an additive, query-independent bias on the decoder's
cross-attention logits, `score(i,t) = q·k/√d + Ω(t)`, built as **trust × evidence + fact**:

    Ω(t) = γ_t · max(ln ε, −Σ_{k∈K_t} ReLU(−ℓ_k))  +  ln(1 − χ_t + ε)
    ℓ_k  = z_k(I) − logsumexp(z_k(B), z_k(O))                 (three-way log-odds of I)
    γ_t  = sg[ 1 − H̄(band around the nearer of {s, τ}) ]      (per-edge trust, stop-grad)
    K_t  = frames between the selected start s and the attended frame t (δ-wide left ramp)
    χ_t  = commit flag (already-emitted frames), UNCONDITIONAL floor, outside γ

Ω(t) ≤ 0 always (m_t ≤ 1): the gate can only DOWN-weight attention, never amplify. Frames the head believes
are inside the target sentence (I wins the majority) get Ω = 0 — the decoder attends them unchanged; frames
the head believes are outside/next-sentence get a γ-scaled penalty.

CRITICAL — the offset the design doc glosses. The doc's F is "what the BIO head reads AND what the decoder
cross-attends". In the real Uni-Sign stack these differ: the BIO head reads the pose-encoder tap (length T);
the decoder cross-attends `mT5_encoder([prompt | pose_tokens])` (length M = prompt_len + T). Ω is defined over
the T pose frames, so `omega_cross_bias` LEFT-PADS prompt_len zeros — the task prompt is never gated — and
aligns Ω to cross-attention columns [prompt_len, prompt_len+T).

Gradient (per the doc): flows only into the softmax logits of contested, above-floor frames in K_t, scaled by
γ_t; nothing flows into γ_t (stop-grad) or χ_t (constant). The selection of (s, τ) is argmax-level and carries
no gradient. Verified against the doc's running-example numbers in tests/test_membership_gate.py.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from data.windowing import BIO

# BIO class indices (data.windowing.BIO = {UNK:0, O:1, B:2, I:3}); UNK is padding-only and excluded from the
# three-way {B,I,O} log-odds and entropy exactly as the doc specifies (ln 3 normalization, 3 classes).
_B, _I, _O = BIO["B"], BIO["I"], BIO["O"]
LN3 = torch.log(torch.tensor(3.0))


@dataclass
class OmegaOutput:
    omega: torch.Tensor         # (B, T) additive bias over the pose frames (Ω ≤ 0)
    logm: torch.Tensor          # (B, T) floored ln(m_t ∨ ε) (before γ)
    gamma: torch.Tensor         # (B, T) per-frame trust γ_t (detached)
    gamma_s: torch.Tensor       # (B,) start-band trust
    gamma_tau: torch.Tensor     # (B,) terminator-band trust (γ_s on the open/forced path)


def three_way_log_odds(bio_logits: torch.Tensor) -> torch.Tensor:
    """ℓ_k = z(I) − logsumexp(z(B), z(O)), the three-way log-odds of I vs {B,O} (doc §2.5, canonical form).

    Pure logit space — no probabilities materialized — so it is the numerically stable path the doc names.
    Gradient split (doc §2.5): ∂ℓ/∂z(I)=1, ∂ℓ/∂z(B)=−P(B|¬I), ∂ℓ/∂z(O)=−P(O|¬I) — the push attacks the actual
    competitor. `bio_logits`: (..., C≥4) raw head logits (UNK ignored).
    """
    zI = bio_logits[..., _I]
    zBO = torch.stack([bio_logits[..., _B], bio_logits[..., _O]], dim=-1)
    return zI - torch.logsumexp(zBO, dim=-1)


def _three_way_entropy(bio_logits: torch.Tensor) -> torch.Tensor:
    # Normalized entropy H̄ = −Σ_c P(c) ln P(c) / ln 3 over the 3 classes {B,I,O} (UNK excluded, renormalized).
    z3 = torch.stack([bio_logits[..., _B], bio_logits[..., _I], bio_logits[..., _O]], dim=-1)
    logp = torch.log_softmax(z3, dim=-1)
    p = logp.exp()
    ln3 = LN3.to(bio_logits.device, bio_logits.dtype)
    return -(p * logp).sum(dim=-1) / ln3  # (...,)


def membership_logm(
    hinge: torch.Tensor, starts: torch.Tensor, ramp: int, eps: float,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directional capped-odds accumulation → floored ln(m_t ∨ ε), doc §2.5 + §2.9 legend.

    `hinge[b,k] = ReLU(−ℓ_k) ≥ 0`. For t ≥ s: ln m_t = −Σ_{s<k≤t} hinge (start s excluded, m_s = 1). For the
    δ-wide left ramp s−δ ≤ t < s: ln m_t = −Σ_{t≤k<s} hinge (doc §2.6). For t < s−δ: the wall (floor ln ε).
    All floored at ln ε (the exact `m_t ∨ ε` form, so the gradient is exactly zero below the doubt cap).

    Fully vectorized via a padded prefix-sum `Cpad[b,j] = Σ_{k<j} hinge[b,k]` (so Cpad[:,1:] == cumsum, and any
    contiguous inclusive sum a..b = Cpad[b+1] − Cpad[a]). Differentiable w.r.t. hinge (→ BIO logits).
    """
    B, T = hinge.shape
    device = hinge.device
    Cpad = F.pad(torch.cumsum(hinge, dim=1), (1, 0))  # (B, T+1): Cpad[b,j] = sum_{k<j} hinge
    s = starts.clamp(0, T - 1).long()
    Cs = Cpad.gather(1, s.view(B, 1))                 # (B,1) = sum_{k<s} hinge
    Cs1 = Cpad.gather(1, (s + 1).view(B, 1))          # (B,1) = sum_{k<=s} hinge

    t = torch.arange(T, device=device).view(1, T).expand(B, T)
    right = Cs1 - Cpad[:, 1:]                          # sum_{0..s} − sum_{0..t}  = −Σ_{s<k≤t}  (valid t≥s)
    left = Cpad[:, :T] - Cs                            # sum_{0..t−1} − sum_{0..s−1} = −Σ_{t≤k<s} (valid t<s)

    s_col = s.view(B, 1)
    logm = torch.where(t >= s_col, right, left)
    logm = torch.where(t < s_col - int(ramp), torch.full_like(logm, float("-inf")), logm)  # wall → floor below
    ln_eps = torch.log(torch.tensor(float(eps), device=device, dtype=logm.dtype))
    logm = torch.maximum(logm, ln_eps)                # exact floor m_t ∨ ε
    if lengths is not None:                            # padded frames never gate real ones; keep them at floor
        valid = t < lengths.view(B, 1)
        logm = torch.where(valid, logm, ln_eps.expand_as(logm))
    return logm


def _band_entropy(
    entropy: torch.Tensor, center: torch.Tensor, half: int, lengths: torch.Tensor,
) -> torch.Tensor:
    # Mean normalized entropy over the ±half band around `center`, clamped to [0, length). (B,)
    B, T = entropy.shape
    device = entropy.device
    t = torch.arange(T, device=device).view(1, T).expand(B, T)
    c = center.view(B, 1)
    band = (t >= c - half) & (t <= c + half) & (t < lengths.view(B, 1)) & (t >= 0)
    band = band & (center.view(B, 1) >= 0)            # no band when the edge does not exist
    w = band.to(entropy.dtype)
    denom = w.sum(dim=1).clamp(min=1.0)
    return (entropy * w).sum(dim=1) / denom


def build_omega(
    bio_logits: torch.Tensor, starts: torch.Tensor, terminators: torch.Tensor,
    commit_mask: torch.Tensor | None = None, lengths: torch.Tensor | None = None,
    delta: int = 3, eps: float = 1e-4, has_terminator: torch.Tensor | None = None,
) -> OmegaOutput:
    """Assemble Ω(t) over the T pose frames (doc §2.9). Returns (B, T); see `omega_cross_bias` for the
    prompt-offset expansion to a cross-attention bias.

    Args:
        bio_logits: (B, T, C≥4) raw BIO head logits.
        starts:      (B,) selected start frame s (argmax-level; no gradient).
        terminators: (B,) terminator frame τ (first O-or-B after s). Use any value where `has_terminator` is
                     False — it is ignored there (open/forced-commit path: γ ≡ γ_s, no right cliff).
        commit_mask: (B, T) bool χ — already-emitted frames get an UNCONDITIONAL ln ε (outside γ). None → zeros.
        lengths:     (B,) real frame counts (excludes right-padding). None → all T valid.
        delta:       δ = ramp width, boundary tolerance, and (as 2δ) trust-band half-width. One constant.
        has_terminator: (B,) bool. None → inferred as (terminators >= 0) & (terminators > starts).
    """
    B, T, _ = bio_logits.shape
    device = bio_logits.device
    if lengths is None: lengths = torch.full((B,), T, dtype=torch.long, device=device)
    lengths = lengths.to(device).long()

    starts = starts.to(device).long()
    terminators = terminators.to(device).long()
    if has_terminator is None: has_terminator = (terminators >= 0) & (terminators > starts)
    has_terminator = has_terminator.to(device).bool()

    # ── evidence: capped-odds membership ──────────────────────────────────────
    ell = three_way_log_odds(bio_logits)              # (B, T)
    hinge = F.relu(-ell)                              # ReLU(−ℓ)
    logm = membership_logm(hinge, starts, ramp=delta, eps=eps, lengths=lengths)

    # ── trust: per-edge γ, stop-gradient (doc §2.8) ───────────────────────────
    entropy = _three_way_entropy(bio_logits).detach()  # sg: γ never receives gradient
    half = 2 * int(delta)
    gamma_s = (1.0 - _band_entropy(entropy, starts, half, lengths)).clamp(0.0, 1.0)          # (B,)
    tau_center = torch.where(has_terminator, terminators, torch.full_like(terminators, -1))
    gamma_tau = (1.0 - _band_entropy(entropy, tau_center, half, lengths)).clamp(0.0, 1.0)
    gamma_tau = torch.where(has_terminator, gamma_tau, gamma_s)  # open/forced: γ ≡ γ_s (doc §2.8 forced path)

    # per-frame attribution by nearest edge (ties → s); γ_s everywhere when no terminator.
    t = torch.arange(T, device=device).view(1, T).expand(B, T)
    near_s = (t - starts.view(B, 1)).abs() <= (t - terminators.view(B, 1)).abs()
    near_s = near_s | (~has_terminator.view(B, 1))
    gamma = torch.where(near_s, gamma_s.view(B, 1), gamma_tau.view(B, 1))  # (B, T), detached

    # ── fact: commit mask, unconditional, OUTSIDE γ (doc §2.7) ────────────────
    omega = gamma * logm
    if commit_mask is not None:
        chi = commit_mask.to(device).float()
        omega = omega + torch.log(1.0 - chi + eps)
    return OmegaOutput(omega=omega, logm=logm, gamma=gamma, gamma_s=gamma_s, gamma_tau=gamma_tau)


class CrossAttnOmegaInjector:
    """Add a fixed, query-independent Ω(t) to a HF encoder-decoder's CROSS-attention via forward pre-hooks —
    so HF's own AR `forward` / `generate` / beam / KV-cache run UNCHANGED and the AR de-risk arm sees the
    identical conditioning as the DLM arm (which injects Ω in its manual decode loop). Adapt HF, don't
    reimplement the decoder. One mechanism, two backbones (verified against transformers 4.57.3 internals):

      T5 / mT5 : the cross-attn folds its mask into `encoder_decoder_position_bias` at block 0 and REUSES it
                 across blocks (modeling_t5.py `T5Attention`: `if position_bias is None: ... += causal_mask`),
                 so adding Ω to each block's cross `attention_mask` kwarg propagates to every layer.
      mBART    : each `MBartDecoderLayer` adds `encoder_attention_mask` to the cross-attn scores directly
                 (`eager_attention_forward`: `attn_weights += attention_mask`; SDPA passes it as `attn_mask`),
                 so adding Ω to `attention_mask` gates every layer.

    Ω is (B,1,1,M) additive (≤0); it broadcasts over heads and decoder queries and is re-applied every decode
    step (fixed conditioning). Beam expands the batch to B·beams — the hook repeats Ω to match. Use via the
    `with_omega(omega_bias)` context so the hooks are inert (identity) whenever no gate is active.
    """
    def __init__(self, lm_model: torch.nn.Module):
        self._omega: torch.Tensor | None = None
        self.handles = [
            m.register_forward_pre_hook(self._pre_hook, with_kwargs=True) 
            for m in self._cross_attn_modules(lm_model)
        ]
        if not self.handles:
            raise ValueError(f"CrossAttnOmegaInjector found no cross-attention modules on {type(lm_model).__name__}")

    @staticmethod
    def _cross_attn_modules(lm_model: torch.nn.Module) -> list[torch.nn.Module]:
        dec = getattr(lm_model, "decoder", None) or getattr(getattr(lm_model, "model", None), "decoder", None)
        if dec is None: return []
        if hasattr(dec, "block"):   # T5 / mT5 stack: block[i].layer[1] == T5LayerCrossAttention
            return [blk.layer[1] for blk in dec.block]
        if hasattr(dec, "layers"):  # mBART: layers[i].encoder_attn == MBartAttention (cross)
            return [layer.encoder_attn for layer in dec.layers]
        return []

    def _pre_hook(self, module, args, kwargs):
        omega = self._omega
        if omega is None: return None  # gate inactive → identity hook
        mask = kwargs.get("attention_mask", None)
        ref = mask if isinstance(mask, torch.Tensor) else omega
        om = omega.to(dtype=ref.dtype, device=ref.device)
        if isinstance(mask, torch.Tensor) and mask.shape[0] != om.shape[0] and mask.shape[0] % om.shape[0] == 0:
            om = om.repeat_interleave(mask.shape[0] // om.shape[0], dim=0)  # beam-expanded batch
        kwargs["attention_mask"] = om if not isinstance(mask, torch.Tensor) else mask + om
        return args, kwargs

    def with_omega(self, omega_bias: torch.Tensor | None):
        injector = self
        class _Ctx:
            def __enter__(self): injector._omega = omega_bias
            def __exit__(self, *exc): injector._omega = None
        return _Ctx()

    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []


def omega_cross_bias(omega: torch.Tensor, memory_len: int, dtype: torch.dtype) -> torch.Tensor:
    """Expand Ω (B, T) over pose frames into a (B, 1, 1, M) additive cross-attention bias.

    LEFT-PADS `prompt_len = M − T` zeros: the task-prompt columns are NEVER gated (Ω=0 → full attention), and
    Ω aligns to the pose columns [prompt_len, M). Broadcasts over heads and decoder queries (query-independent —
    the doc's non-monotonicity guarantee, §2.10). Add this to the existing cross-attention mask bias.
    """
    B, T = omega.shape
    prompt_len = int(memory_len) - int(T)
    if prompt_len < 0:
        raise ValueError(f"memory_len ({memory_len}) < pose frames ({T}); Ω cannot be aligned to cross-attn columns")
    full = F.pad(omega, (prompt_len, 0), value=0.0).to(dtype)   # (B, M); prompt columns = 0
    return full.view(B, 1, 1, int(memory_len))


if __name__ == "__main__": # Self-check against docs/membership_gate.md running example (§2.9 table / Appendix C).
    torch.manual_seed(0)

    def frame(pB, pI, pO):  # probabilities → logits (log; softmax recovers them up to a constant)
        return torch.log(torch.tensor([0.0, pO, pB, pI]).clamp_min(1e-9))  # [UNK,O,B,I]

    # S2 = [s=1 .. τ=4]; frame 0 = gap (pre-start), 1-3 interior/pre-boundary, 4 = B opening S3, 5 = S3 interior.
    rows = [frame(0.05, 0.15, 0.80),   # 0 gap (r=0.15/0.85 → hinge 1.734)
            frame(0.005, 0.99, 0.005), # 1 s: interior (hinge 0)
            frame(0.01, 0.97, 0.02),   # 2 interior (hinge 0)
            frame(0.05, 0.15, 0.80),   # 3 pre-boundary (hinge 1.734)
            frame(0.90, 0.06, 0.04),   # 4 τ: B opens S3 (hinge 2.75)
            frame(0.02, 0.90, 0.08)]   # 5 S3 interior (hinge 0, product held)
    z = torch.stack(rows).unsqueeze(0)  # (1, 6, 4)
    out = build_omega(z, starts=torch.tensor([1]), terminators=torch.tensor([4]), delta=3, eps=1e-4)

    ell = three_way_log_odds(z)[0]
    hinge = F.relu(-ell)
    assert abs(hinge[3].item() - 1.734) < 0.01, hinge
    assert abs(hinge[4].item() - 2.75) < 0.02, hinge
    assert hinge[1].item() < 1e-3 and hinge[5].item() < 1e-3  # interiors contribute 0

    logm = out.logm[0]
    assert abs(logm[1].item()) < 1e-4                                   # m_s = 1
    assert abs(logm[3].item() - (-1.734)) < 0.01                        # pre-boundary: itself
    assert abs(logm[4].item() - (-(1.734 + 2.75))) < 0.02              # B: pre-boundary + B
    assert abs(logm[5].item() - logm[4].item()) < 1e-4                 # S3 interior: product held (no rise)
    # left ramp at frame 0 (s−1): one gap frame hinge 1.734
    assert abs(logm[0].item() - (-1.734)) < 0.01, logm

    # gradient reaches BIO logits at contested frames, zero at satisfied interiors
    z2 = z.clone().requires_grad_(True)
    o2 = build_omega(z2, starts=torch.tensor([1]), terminators=torch.tensor([4]), delta=3, eps=1e-4)
    o2.omega.sum().backward()
    assert z2.grad[0, 4].abs().sum() > 0, "contested frame must get gradient"
    assert z2.grad[0, 2].abs().sum() < 1e-6, "satisfied interior must get no gradient"
    print("membership_gate self-check OK:",
          f"logm={logm.tolist()} gamma_s={out.gamma_s.item():.3f} gamma_tau={out.gamma_tau.item():.3f}")
