"""Faithful Moryossef 2026 external segmenter for calibration and the RQ2 cascade.

Raw keypoints (+velocity) → UNet CNN → RoPE Transformer → phrase BIO head (moryossef26/model.py). An INDEPENDENT
model, deliberately a different input space from the in-system FSM head (train.bio_pretrain, train-bio) — see
moryossef26/trainer.py and docs/membership_gate.md §1.4.
"""
