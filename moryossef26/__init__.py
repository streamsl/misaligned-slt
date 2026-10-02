"""Faithful Moryossef 2026 external segmenter for the RQ2 cascade.

Raw keypoints (+velocity) → UNet CNN → RoPE Transformer → phrase BIO head (moryossef26/model.py). An INDEPENDENT model 
with their own input contract (keypoints, normalization, velocity). Chunking, augmentation, labels, dev monitor and 
best-epoch floor are S1's (data.chunks, train.helpers.evaluate_bio_chunks) — see docs/membership_gate.md §4.
"""
