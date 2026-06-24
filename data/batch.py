from __future__ import annotations
from data.windowing import BIO
import torch


def collate_windows(batch: list[dict], visual_padding: str = "none") -> dict[str, torch.Tensor | list]:
    prepared = []
    for item in batch:
        poses_i = torch.as_tensor(item["poses"]).float()
        ts_i = torch.as_tensor(item["timestamps_s"]).float()
        labels_i = torch.as_tensor(item["bio_labels"]).long()
        # Uni-Sign uses raw windows; padding is handled at batch level + masked via frame_mask (no boundary halos).
        if visual_padding in {"none", "zero"}: mask_i = torch.ones(poses_i.shape[0], dtype=torch.bool)
        else: raise ValueError(f"Unsupported visual_padding={visual_padding!r} (Uni-Sign uses 'none')")
        prepared.append((item, poses_i, ts_i, mask_i, labels_i))

    max_len = max(poses_i.shape[0] for _, poses_i, _, _, _ in prepared)
    pose_shape = prepared[0][1].shape[1:]
    poses, frame_masks, timestamps  = [], [], []
    bio_labels, specs, targets, anchor_spans = [], [], [], []

    for item, poses_i, ts_i, mask_i, labels_i in prepared:
        n = poses_i.shape[0]
        pad = max_len - n
        poses.append(torch.nn.functional.pad(poses_i, (0, 0, 0, 0, 0, pad)))
        frame_masks.append(torch.cat([mask_i, torch.zeros(pad, dtype=torch.bool)]))
        timestamps.append(torch.nn.functional.pad(ts_i, (0, pad)))
        bio_labels.append(torch.cat([labels_i, torch.full((pad,), BIO["UNK"], dtype=torch.long)]))
        specs.append(item["spec"])
        targets.append(item.get("translation_target"))
        anchor_spans.append(item.get("anchor_span"))

    return {
        "poses": torch.stack(poses).reshape(len(batch), max_len, *pose_shape),
        "frame_mask": torch.stack(frame_masks), "timestamps_s": torch.stack(timestamps),
        "bio_labels": torch.stack(bio_labels), "specs": specs,
        "translation_targets": targets, "anchor_spans": anchor_spans,
    }


class WindowCollator:# Collate windows and optionally tokenize complete-anchor references.
    def __init__(
        self, tokenizer=None, max_text_tokens: int = 128,
        pad_to_max_length: bool = True, visual_padding: str = "none",
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = int(max_text_tokens)
        self.pad_to_max_length = bool(pad_to_max_length)
        self.visual_padding = str(visual_padding)

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None: raise ValueError("WindowCollator tokenization requested without a tokenizer")
        padding = "max_length" if self.pad_to_max_length else True
        encoded = self.tokenizer(
            texts, padding=padding, truncation=True,
            max_length=self.max_text_tokens, return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor | list | dict]:
        out = collate_windows(batch, visual_padding=self.visual_padding)
        out["mode_names"] = [spec["mode"] if isinstance(spec, dict) else spec.mode for spec in out["specs"]]
        out["mode2_subcases"] = [(spec.get("subcase") if isinstance(spec, dict) else spec.subcase) for spec in out["specs"]]
        out["translation_supervised"] = torch.tensor(
            [target is not None for target in out["translation_targets"]],
            dtype=torch.bool,
        )
        out["confidence_bound_candidates"] = torch.tensor([
            mode == "mode2" and subcase == "right" and anchor is not None
            for mode, subcase, anchor in zip(out["mode_names"], out["mode2_subcases"], out["anchor_spans"])
        ], dtype=torch.bool)

        if self.tokenizer is not None:
            target_texts = [
                target["text"] if isinstance(target, dict) else (target.text if target is not None else "")
                for target in out["translation_targets"]
            ]
            reference_texts = [
                anchor["text"] if isinstance(anchor, dict) else (anchor.text if anchor is not None else "")
                for anchor in out["anchor_spans"]
            ]
            target_tokens = self._tokenize_texts(target_texts)
            reference_tokens = self._tokenize_texts(reference_texts)
            target_tokens["labels"][~out["translation_supervised"]] = -100
            out["target_tokens"] = target_tokens
            out["reference_tokens"] = reference_tokens

        full_items = [item["full_evidence"] for item in batch if item.get("full_evidence") is not None]
        full_indices = [idx for idx, item in enumerate(batch) if item.get("full_evidence") is not None]
        out["full_evidence_indices"] = torch.tensor(full_indices, dtype=torch.long)
        out["full_evidence"] = collate_windows(full_items, visual_padding=self.visual_padding) if full_items else None
        return out
