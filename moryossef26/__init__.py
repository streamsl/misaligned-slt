"""Faithful Moryossef 2026 external segmenter for the cascade rows.

Raw keypoints (+velocity) → UNet CNN → RoPE Transformer → phrase BIO head (moryossef26/model.py). An INDEPENDENT model 
with their own input contract (keypoints, normalization, velocity). Chunking, augmentation, labels, dev monitor and 
best-epoch floor are S1's (data.chunks, train.helpers.evaluate_segmenter). It trains on plain chunks only: S1's windows 
and restart chunks (bio_pretrain.yaml pretrain.inputs) are not its data.
"""
