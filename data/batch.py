from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from data.windowing import BIO


def frame_mask_for(n_frames: int, visual_padding: str = "none") -> torch.Tensor:
    """All-True per-frame mask for an unpadded window. Uni-Sign uses raw windows ('none') and has no 
    zero-pad mode at all (datasets.py collate_fn repeats the last frame + masks by true length)."""
    if visual_padding == "none": return torch.ones(int(n_frames), dtype=torch.bool)
    raise ValueError(f"Unsupported visual_padding={visual_padding!r} (Uni-Sign uses 'none')")


def repeat_last_frame(poses: torch.Tensor, pad: int) -> torch.Tensor:
    """Right-pad a (T, ...) pose tensor by `pad` frames, REPEATING THE LAST FRAME (Uni-Sign Base_Dataset.collate_fn), 
    not zeros: the pose branch's temporal GCN (kernel 5) has no mask, so zero pads leak into the last real frames' 
    features before the LM attention mask drops them, making a row depend on its batchmates' lengths. Repeating keeps 
    the receptive-field edge batch-invariant."""
    if pad <= 0: return poses
    if poses.shape[0]: return torch.cat([poses, poses[-1:].expand(pad, *poses.shape[1:])])
    return torch.nn.functional.pad(poses, (0,) * (2 * (poses.ndim - 1)) + (0, pad))


def collate_windows(batch: list[dict], visual_padding: str = "none") -> dict[str, torch.Tensor | list]:
    prepared = []
    for item in batch:
        poses_i = torch.as_tensor(item["poses"]).float()
        ts_i = torch.as_tensor(item["timestamps_s"]).float()
        labels_i = torch.as_tensor(item["bio_labels"]).long()
        mask_i = frame_mask_for(poses_i.shape[0], visual_padding)
        prepared.append((item, poses_i, ts_i, mask_i, labels_i))

    max_len = max(poses_i.shape[0] for _, poses_i, _, _, _ in prepared)
    pose_shape = prepared[0][1].shape[1:]
    poses, frame_masks, timestamps  = [], [], []
    bio_labels, specs, targets, anchor_spans = [], [], [], []
    commit_masks = []

    for item, poses_i, ts_i, mask_i, labels_i in prepared:
        n = poses_i.shape[0]
        pad = max_len - n
        chi_i = item.get("commit_mask")
        chi_t = torch.zeros(n, dtype=torch.bool) if chi_i is None else torch.as_tensor(chi_i).bool()
        commit_masks.append(torch.cat([chi_t, torch.zeros(pad, dtype=torch.bool)]))  # padding is never "committed"
        # Padded frames stay masked (attention) and UNK (BIO loss); only the GCN receptive-field edge changes.
        poses.append(repeat_last_frame(poses_i, pad))
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
        "commit_mask": torch.stack(commit_masks),
        "translation_targets": targets, "anchor_spans": anchor_spans,
    }


class WindowCollator:# Collate windows and optionally tokenize complete-anchor references.
    def __init__(
        self, tokenizer=None, max_text_tokens: int = 128, pad_to_max_length: bool = True, 
        visual_padding: str = "none", block_size: int = 1, eos_supervision_tokens: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_text_tokens = int(max_text_tokens)
        self.pad_to_max_length = bool(pad_to_max_length)
        self.visual_padding = str(visual_padding)
        # Dynamic padding (pad_to_max_length=False) needs the DLM's canvas geometry to stay equivalent to a
        # full-width canvas: see _tokenize_texts.
        self.block_size = max(1, int(block_size))
        self.eos_supervision_tokens = max(0, int(eos_supervision_tokens))

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        if self.tokenizer is None: raise ValueError("WindowCollator tokenization requested without a tokenizer")
        # Every decoder forward runs over the full canvas, so padding to max_text_tokens costs compute on padding:
        # captions are ~15 tokens against a 128 canvas. Dynamic padding sizes it to the batch instead, with two
        # slots of headroom that the DLM path requires and batch-max alone would not leave:
        #   1. supervise_trailing_eos needs eos_supervision_tokens PAD slots after the longest row's sentence;
        #   2. the confidence-bound reference shift (models/streaming_slt.py) drops the last column, which must be
        #      PAD or the longest row loses its real final token from the gate.
        # Block alignment matters too — BD3LM attends bidirectionally WITHIN a block, so a canvas ending mid-block
        # would change the last positions' logits. With headroom + alignment the result matches the full canvas.
        encoded = self.tokenizer(
            texts, padding="max_length" if self.pad_to_max_length else True,
            truncation=True, max_length=self.max_text_tokens, return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if not self.pad_to_max_length:
            need = input_ids.shape[1] + 1 + self.eos_supervision_tokens
            width = min(self.max_text_tokens, math.ceil(need / self.block_size) * self.block_size)
            if width > input_ids.shape[1]:
                pad_id = int(self.tokenizer.pad_token_id)
                grow = width - input_ids.shape[1]
                input_ids = F.pad(input_ids, (0, grow), value=pad_id)
                attention_mask = F.pad(attention_mask, (0, grow), value=0)
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
