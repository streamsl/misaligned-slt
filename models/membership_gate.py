"""Membership gate Ω(t): additive, query-independent bias on the decoder's cross-attention logits,
`score(i,t) = q·k/√d + Ω(t)`, conditioning translation on the BIO head's segmentation beliefs.

`docs/membership_gate.md` §2.9, as trust × evidence + fact:

    Ω(t) = γ_t · max(ln ε, −Σ_{k∈K_t} ReLU(−ℓ_k))  +  ln(1 − χ_t + ε)
    ℓ_k  = z_k(I) − logsumexp(z_k(B), z_k(O))                 (three-way log-odds of I)
    γ_t  = sg[ 1 − H̄(band around the nearer of {s, τ}) ]      (per-edge trust, stop-grad)
    K_t  = frames between the selected start s and the attended frame t (δ-wide left ramp)
    χ_t  = commit flag (already-emitted frames), unconditional floor, outside γ

Ω ≤ 0 always: confident-I frames get Ω = 0, outside ones a γ-scaled penalty.

Offset the doc glosses: the BIO head reads the pose tap (T) but the decoder cross-attends
`mT5_encoder([prompt | pose_tokens])` (M = prompt_len + T) — see `omega_cross_bias`.

Gradient reaches only contested, above-floor frames in K_t, scaled by γ_t; γ_t (stop-grad), χ_t and 
the argmax selection of (s, τ) carry none.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from data.windowing import BIO

# data.windowing.BIO = {UNK:0, O:1, B:2, I:3}. UNK is padding-only: excluded from the {B,I,O} log-odds 
# and entropy (hence the ln 3 normalization).
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
    """ℓ_k = z(I) − logsumexp(z(B), z(O)), log-odds of I vs {B,O} (doc §2.5). `bio_logits`: (..., C≥4) raw logits.

    Pure logit space for stability. ∂ℓ/∂z(I)=1, ∂ℓ/∂z(B)=−P(B|¬I), ∂ℓ/∂z(O)=−P(O|¬I) — 
    the push attacks the actual competitor.
    """
    zI = bio_logits[..., _I]
    zBO = torch.stack([bio_logits[..., _B], bio_logits[..., _O]], dim=-1)
    return zI - torch.logsumexp(zBO, dim=-1)


def _three_way_entropy(bio_logits: torch.Tensor) -> torch.Tensor:
    # Normalized entropy H̄ = −Σ_c P(c) ln P(c) / ln 3 over {B,I,O}.
    z3 = torch.stack([bio_logits[..., _B], bio_logits[..., _I], bio_logits[..., _O]], dim=-1)
    logp = torch.log_softmax(z3, dim=-1)
    p = logp.exp()
    ln3 = LN3.to(bio_logits.device, bio_logits.dtype)
    return -(p * logp).sum(dim=-1) / ln3  # (...,)


def membership_logm(
    hinge: torch.Tensor, starts: torch.Tensor, ramp: int, eps: float,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Directional capped-odds accumulation → floored ln(m_t ∨ ε), doc §2.5–§2.6. Differentiable w.r.t. hinge.

    `hinge[b,k] = ReLU(−ℓ_k) ≥ 0`. t ≥ s: ln m_t = −Σ_{s<k≤t} hinge (s excluded, m_s = 1); δ-wide left ramp
    s−δ ≤ t < s: −Σ_{t≤k<s} hinge; t < s−δ: the wall. Floored at ln ε (exact `m_t ∨ ε` → gradient exactly 0
    below the doubt cap). Prefix sums `Cpad[b,j] = Σ_{k<j} hinge`; inclusive a..b = Cpad[b+1] − Cpad[a].
    """
    B, T = hinge.shape
    device = hinge.device
    in_dtype, hinge = hinge.dtype, hinge.float()
    Cpad = F.pad(torch.cumsum(hinge, dim=1), (1, 0))  # (B, T+1): Cpad[b,j] = sum_{k<j} hinge
    s = starts.clamp(0, T - 1).long()
    Cs = Cpad.gather(1, s.view(B, 1))                 # (B,1) = sum_{k<s} hinge
    Cs1 = Cpad.gather(1, (s + 1).view(B, 1))          # (B,1) = sum_{k<=s} hinge

    t = torch.arange(T, device=device).view(1, T).expand(B, T)
    right = Cs1 - Cpad[:, 1:]                         # sum_{0..s} − sum_{0..t}  = −Σ_{s<k≤t}  (valid t≥s)
    left = Cpad[:, :T] - Cs                           # sum_{0..t−1} − sum_{0..s−1} = −Σ_{t≤k<s} (valid t<s)

    s_col = s.view(B, 1)
    logm = torch.where(t >= s_col, right, left)
    logm = torch.where(t < s_col - int(ramp), torch.full_like(logm, float("-inf")), logm)  # wall → floor below
    ln_eps = torch.log(torch.tensor(float(eps), device=device, dtype=logm.dtype))
    logm = torch.maximum(logm, ln_eps)                # exact floor m_t ∨ ε
    if lengths is not None:                           # padded frames never gate real ones; keep them at floor
        valid = t < lengths.view(B, 1)
        logm = torch.where(valid, logm, ln_eps.expand_as(logm))
    return logm.to(in_dtype)


def _band_entropy(entropy: torch.Tensor, center: torch.Tensor, half: int, lengths: torch.Tensor) -> torch.Tensor:
    # Mean H̄ over the ±half band around `center`, clamped to [0, length). (B,)
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
    """Assemble Ω(t) over the T pose frames (doc §2.9) → (B, T); `omega_cross_bias` expands it to a cross-attn bias.

    Args:
        bio_logits: (B, T, C≥4) raw BIO head logits.
        starts:      (B,) start frame s (argmax-level; no gradient).
        terminators: (B,) terminator τ (first O-or-B after s); ignored where `has_terminator` is False
                     (open/forced-commit path: γ ≡ γ_s, no right cliff).
        commit_mask: (B, T) bool χ — emitted frames get an unconditional ln ε (outside γ). None → zeros.
        lengths:     (B,) real frame counts (excludes right-padding). None → all T valid.
        delta:       δ = ramp width, boundary tolerance, and (as 2δ) trust-band half-width. One constant.
        has_terminator: (B,) bool. None → (terminators >= 0) & (terminators > starts).
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

    # ── right wall (mirrors the left wall at s−δ): frames past τ+δ are non-members ──
    # Membership only decays where hinge > 0, i.e. at a gap; a back-to-back successor is confident-I throughout,
    # so without this wall Ω ≈ 0 past τ and the gate cannot mask the neighbour. No-op for gap-terminated spans,
    # inert on the open/forced path; hard floor → gradient exactly 0 below it.
    right_wall = has_terminator.view(B, 1) & (t > terminators.view(B, 1) + int(delta))
    ln_eps = torch.log(torch.tensor(float(eps), device=device, dtype=logm.dtype))
    logm = torch.where(right_wall, ln_eps.expand_as(logm), logm)

    # ── fact: commit mask, unconditional, OUTSIDE γ (doc §2.7) ────────────────
    omega = gamma * logm
    if commit_mask is not None:
        chi = commit_mask.to(device).float()
        omega = omega + torch.log(1.0 - chi + eps)
    return OmegaOutput(omega=omega, logm=logm, gamma=gamma, gamma_s=gamma_s, gamma_tau=gamma_tau)


class CrossAttnOmegaInjector:
    """Add Ω(t) to a HF encoder-decoder's cross-attention via forward pre-hooks, so HF's `forward` / `generate` /
    beam / KV-cache run unchanged and the AR arm sees what the DLM arm injects in its manual decode loop.
    Verified against transformers 4.57.3 internals:

      T5 / mT5 : cross-attn folds its mask into `encoder_decoder_position_bias` at block 0 and reuses it across
                 blocks, so adding Ω to each block's cross `attention_mask` kwarg reaches every layer.
      mBART    : each `MBartDecoderLayer` adds `encoder_attention_mask` to the cross-attn scores directly
                 (eager `attn_weights += attention_mask`; SDPA `attn_mask`), so Ω gates every layer.

    Ω is (B,1,1,M), broadcasts over heads and queries, re-applied every decode step. Beam expands the batch to
    B·beams — the hook repeats Ω to match. Use via `with_omega(omega_bias)`, inert when no gate is active.
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
        # Accepts a full HF encoder-decoder OR a bare decoder stack (T5Stack / MBartDecoder) — the DLM's
        # prefix-KV-cache path holds only the stack, and Ω injection is identical either way.
        dec = getattr(lm_model, "decoder", None) or getattr(getattr(lm_model, "model", None), "decoder", None) or lm_model
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
        # Beam search expands the batch to B·beams. Read the target batch from the mask when there is one, else
        # from the query states: keying expansion off the mask ALONE left Ω un-expanded on any backbone that
        # cross-attends without a mask, gating beam rows with another row's Ω.
        hidden = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
        tgt = mask.shape[0] if isinstance(mask, torch.Tensor) else (
            hidden.shape[0] if isinstance(hidden, torch.Tensor) else om.shape[0])
        if tgt != om.shape[0] and om.shape[0] and tgt % om.shape[0] == 0:
            om = om.repeat_interleave(tgt // om.shape[0], dim=0)
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
    """Expand Ω (B, T) into a (B, 1, 1, M) additive bias for the existing cross-attention mask bias.

    Left-pads `prompt_len = M − T` zeros: the prompt is never gated, Ω aligns to pose columns [prompt_len, M)
    and broadcasts over heads and queries (query-independent — doc §2.10).
    """
    B, T = omega.shape
    prompt_len = int(memory_len) - int(T)
    if prompt_len < 0:
        raise ValueError(f"memory_len ({memory_len}) < pose frames ({T}); Ω cannot be aligned to cross-attn columns")
    full = F.pad(omega, (prompt_len, 0), value=0.0).to(dtype)   # (B, M); prompt columns = 0
    return full.view(B, 1, 1, int(memory_len))


if __name__ == "__main__": # Self-check vs the docs/membership_gate.md running example (§2.9 / Appendix C).
    torch.manual_seed(0)

    def frame(pB, pI, pO):  # probabilities → logits (softmax recovers them up to a constant)
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
