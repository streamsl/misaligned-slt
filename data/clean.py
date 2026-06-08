from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import torch
from torch.utils.data import Dataset
from data.gfslt_padding import pad_visual_sequence_gfslt
from data.loader import VideoRecord
from data.windowing import SentenceSpan
from poses import load_pose_window


def noise_inject(
    text: str, mask_token: str, rng: np.random.Generator,
    noise_rate: float = 0.15, noise_type: str = "omit_last", random_shuffle: bool = False,
) -> str:
    """Word-level text corruption for GFSLT-VLP CMLM pretraining.

    Faithful port of GFSLT-VLP utils.NoiseInjecting (github.com/zhoubenjia/
    GFSLT-VLP). `omit_last` masks a random trailing fraction (up to noise_rate)
    of words with the mask token; `omit` masks a random (1 - noise_rate)-kept
    subset. Whole words are replaced by the literal mask token before tokenizing,
    so each becomes the tokenizer's <mask> id.
    """
    words = text.split()
    if not words: return text
    if noise_type == "omit_last":
        keep = len(words) - int(np.ceil(len(words) * float(rng.uniform(0.0, noise_rate))))
        noised = [w if i < keep else mask_token for i, w in enumerate(words)]
    elif noise_type == "omit":
        n_keep = int(len(words) * (1.0 - noise_rate))
        keep_idx = set(rng.choice(len(words), size=max(0, n_keep), replace=False).tolist())
        noised = [w if i in keep_idx else mask_token for i, w in enumerate(words)]
    else: raise ValueError(f"Unsupported noise_type={noise_type}")
    if random_shuffle and rng.uniform(0.0, 1.0) > 0.5: rng.shuffle(noised)
    return " ".join(noised)

@dataclass(frozen=True)
class CleanSentenceItem:
    video_id: str
    span: SentenceSpan

class CleanSentenceDataset(Dataset): # GT-trimmed sentence clips for Stage 1 VLP and the clean AR baseline.
    def __init__(self, records: list[VideoRecord], max_items: int | None = None):
        self.records_by_id = {record.video_id: record for record in records}
        items = [
            CleanSentenceItem(video_id=record.video_id, span=span)
            for record in records for span in record.sentences
        ]
        self.items = items[: int(max_items)] if max_items is not None else items
        if not self.items: raise ValueError("CleanSentenceDataset requires at least one sentence span")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[int(index)]
        record = self.records_by_id[item.video_id]
        poses, abs_timestamps = load_pose_window(record.pose, item.span.start_s, item.span.end_s, normalize=True)
        return {
            "poses": poses, "timestamps_s": abs_timestamps - item.span.start_s,
            "frame_mask": np.ones((poses.shape[0],), dtype=bool), "text": item.span.text,
            "video_id": item.video_id, "start_s": item.span.start_s, "end_s": item.span.end_s,
        }

class CleanSentenceCollator:
    def __init__(
        self, tokenizer=None, max_text_tokens: int = 128, visual_padding: str = "gfslt", cmlm: bool = False,
        noise_rate: float = 0.15, noise_type: str = "omit_last", random_shuffle: bool = False, seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = int(max_text_tokens)
        self.visual_padding = str(visual_padding)
        self.cmlm = bool(cmlm)
        self.noise_rate = float(noise_rate)
        self.noise_type = str(noise_type)
        self.random_shuffle = bool(random_shuffle)
        self._rng = np.random.default_rng(int(seed))
        self._mask_token = getattr(tokenizer, "mask_token", None) or "<mask>"

    def __call__(self, batch: list[dict]) -> dict:
        prepared = []
        for item in batch:
            poses_i = torch.as_tensor(item["poses"]).float()
            ts_i = torch.as_tensor(item["timestamps_s"]).float()
            if self.visual_padding == "gfslt": poses_i, ts_i, mask_i, _ = pad_visual_sequence_gfslt(poses_i, ts_i)
            elif self.visual_padding in {"none", "zero"}: mask_i = torch.ones(poses_i.shape[0], dtype=torch.bool)
            else: raise ValueError(f"Unsupported visual_padding={self.visual_padding}")
            prepared.append((item, poses_i, ts_i, mask_i))

        max_len = max(poses_i.shape[0] for _, poses_i, _, _ in prepared)
        pose_shape = prepared[0][1].shape[1:]
        poses, masks, timestamps, texts, meta = [], [], [], [], []

        for item, poses_i, ts_i, mask_i in prepared:
            n = poses_i.shape[0]
            pad = max_len - n
            poses.append(torch.nn.functional.pad(poses_i, (0, 0, 0, 0, 0, pad)))
            masks.append(torch.cat([mask_i, torch.zeros(pad, dtype=torch.bool)]))
            timestamps.append(torch.nn.functional.pad(ts_i, (0, pad)))
            texts.append(item["text"])
            meta.append({k: item[k] for k in ("video_id", "start_s", "end_s")})
        out = {
            "poses": torch.stack(poses).reshape(len(batch), max_len, *pose_shape),
            "frame_mask": torch.stack(masks), "timestamps_s": torch.stack(timestamps),
            "texts": texts, "meta": meta,
        }
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts, padding="max_length", truncation=True,
                max_length=self.max_text_tokens, return_tensors="pt",
            )
            labels = encoded["input_ids"].clone()
            labels[encoded["attention_mask"] == 0] = -100
            out["text_tokens"] = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels,
            }
            if self.cmlm:
                masked_texts = [noise_inject(
                    t, mask_token=self._mask_token, rng=self._rng,
                    noise_rate=self.noise_rate, noise_type=self.noise_type,
                    random_shuffle=self.random_shuffle,
                ) for t in texts]
                masked_encoded = self.tokenizer(
                    masked_texts, padding="max_length", truncation=True,
                    max_length=self.max_text_tokens, return_tensors="pt",
                )
                out["masked_text_tokens"] = {
                    "input_ids": masked_encoded["input_ids"],
                    "attention_mask": masked_encoded["attention_mask"],
                }
        return out
