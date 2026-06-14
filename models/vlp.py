from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from models.gfslt import GFSLTConfig, GFSLTVisualBackbone, load_gfslt_mbart


@dataclass
class VLPOutput:
    loss: torch.Tensor
    image_features: torch.Tensor
    text_features: torch.Tensor
    logits_per_image: torch.Tensor
    logits_per_text: torch.Tensor
    contrastive_loss: torch.Tensor | None = None
    cmlm_loss: torch.Tensor | None = None


class PoseTextCLIP(nn.Module):
    """GFSLT-VLP stage-1 pretraining for pose clips and text.

    Two objectives, exactly as GFSLT-VLP train_vlp_v2.py (arXiv 2307.14768):
      1. Sign-Text contrastive alignment (SLRCLIP): aligns the visual-encoder and
         text-encoder spaces with a symmetric InfoNCE loss.
      2. Conditional masked LM (Text_Decoder): the mBART *decoder* reconstructs the clean sentence from the text 
         encoder's representation of a noise-injected sentence. This is the half that pretrains the decoder reused 
         at stage 2; the text encoder is detached here so CMLM trains only decoder + lm_head.

    The CMLM decoder is `self.visual.mbart`'s decoder, so the pretrained decoder
    travels with the saved visual backbone into both downstream arms.
    """
    def __init__(
        self, config: GFSLTConfig, projection_dim: int = 1024, logit_scale_init: float = 0.07,
        cmlm_lambda: float = 1.0, cmlm_label_smoothing: float = 0.2, label_temp: float = 10.0,
    ):
        super().__init__()
        # GFSLT-VLP SLRCLIP target temperature: utils.KLLoss does softmax(label * 10) on the eye target.
        self.label_temp = float(label_temp)
        self.visual = GFSLTVisualBackbone(config)
        # Text encoder is a second instance of the same trimmed mBART (separate weights). GFSLT-VLP
        # used the full trimmed model here; we use the depth-trimmed one everywhere for speed.
        self.text_mbart = load_gfslt_mbart(config.mbart_name)
        d_model = self.visual.mbart.config.d_model

        self.visual_cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02) # CLS token for aggregation
        self.image_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.text_proj = nn.Linear(self.text_mbart.config.d_model, projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1.0 / logit_scale_init)))

        self.pad_token_id = int(self.visual.mbart.config.pad_token_id)
        self.cmlm_loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=float(cmlm_label_smoothing))
        self.cmlm_lambda = float(cmlm_lambda)


    def encode_image(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        timestamps_s: torch.Tensor | None = None,
    ) -> torch.Tensor:
        post_vlp, mask, _ = self.visual.extract_post_vlp(poses, frame_mask, timestamps_s)
        batch = post_vlp.shape[0]
        cls_token = self.visual_cls.expand(batch, -1, -1)
        inputs = torch.cat([cls_token, post_vlp], dim=1)
        attention_mask = torch.cat([torch.ones(batch, 1, dtype=torch.long, device=post_vlp.device), mask.long()], dim=1)
        out = self.visual.mbart.model.encoder(inputs_embeds=inputs, attention_mask=attention_mask, return_dict=True)
        return F.normalize(self.image_proj(out.last_hidden_state[:, 0]), dim=-1)


    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_mbart.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        # GFSLT-VLP pools at input_ids.argmax(dim=-1), which selects the
        # high-id mBART language-code token in standard mBART-formatted text.
        pooled_indices = input_ids.argmax(dim=-1) # Find EOS token position
        invalid = attention_mask.gather(1, pooled_indices.unsqueeze(1)).squeeze(1) == 0
        if invalid.any():
            fallback = attention_mask.long().sum(dim=1).clamp(min=1) - 1
            pooled_indices = torch.where(invalid, fallback, pooled_indices)
        pooled = out.last_hidden_state[torch.arange(input_ids.shape[0], device=input_ids.device), pooled_indices]
        return F.normalize(self.text_proj(pooled), dim=-1)


    def cmlm_loss(
        self, clean_input_ids: torch.Tensor, clean_attention_mask: torch.Tensor,
        masked_input_ids: torch.Tensor, masked_attention_mask: torch.Tensor,
    ) -> torch.Tensor: # GFSLT-VLP Text_Decoder CMLM loss; text encoder detached (decoder-only).
        with torch.no_grad():
            encoder_hidden = self.text_mbart.model.encoder(
                input_ids=masked_input_ids,
                attention_mask=masked_attention_mask,
                return_dict=True,
            ).last_hidden_state
        decoder_input_ids = shift_tokens_right(clean_input_ids, self.pad_token_id)
        decoder_out = self.visual.mbart.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=clean_attention_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=masked_attention_mask,
            return_dict=True,
        )
        lm_logits = self.visual.mbart.lm_head(decoder_out.last_hidden_state) + self.visual.mbart.final_logits_bias
        return self.cmlm_loss_fct(lm_logits.reshape(-1, lm_logits.shape[-1]), clean_input_ids.reshape(-1))


    def forward(
        self, poses: torch.Tensor, frame_mask: torch.Tensor,
        text_input_ids: torch.Tensor, text_attention_mask: torch.Tensor, timestamps_s: torch.Tensor | None = None,
        masked_text_input_ids: torch.Tensor | None = None, masked_text_attention_mask: torch.Tensor | None = None,
    ) -> VLPOutput:
        image_features = self.encode_image(poses, frame_mask, timestamps_s=timestamps_s)
        text_features = self.encode_text(text_input_ids, text_attention_mask)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits_i = scale * image_features @ text_features.t()
        logits_t = logits_i.t()

        # GFSLT-VLP SLRCLIP loss (utils.KLLoss): KL( softmax(logits) || softmax(eye * label_temp) ),
        # symmetric over image/text directions. reduction="batchmean" reproduces GFSLT's
        # KLDivLoss(size_average=True) * batch_size (= sum/N). The softmax(eye*10) target is near-one-hot
        # (~0.003 smoothing); with one-hot this equals cross-entropy, so this is the faithful GFSLT form.
        n = logits_i.shape[0]
        eye = torch.eye(n, dtype=logits_i.dtype, device=logits_i.device)
        # Duplicate captions are FALSE negatives (measured 7.5% exact duplicates on Auslan; PHOENIX
        # weather text is also highly repetitive). Fold them in as ADDITIONAL positives (multi-positive
        # target) rather than pushing them apart — and unlike masking logits to -inf this keeps the KL
        # target all-positive/finite (no 0*-inf NaN). No duplicates ⇒ exactly softmax(eye*label_temp).
        same_text = (text_input_ids.unsqueeze(0) == text_input_ids.unsqueeze(1)).all(dim=-1)
        positives = (eye + same_text.to(logits_i.dtype)).clamp(max=1.0)  # 1 on diagonal OR identical-text pairs
        target = F.softmax(positives * self.label_temp, dim=1)
        contrastive = (
            F.kl_div(F.log_softmax(logits_i, dim=1), target, reduction="batchmean")
            + F.kl_div(F.log_softmax(logits_t, dim=1), target.t(), reduction="batchmean")
        ) / 2.0

        cmlm, loss = None, contrastive
        if self.cmlm_lambda > 0.0 and masked_text_input_ids is not None:
            cmlm = self.cmlm_loss(
                clean_input_ids=text_input_ids, clean_attention_mask=text_attention_mask,
                masked_input_ids=masked_text_input_ids, masked_attention_mask=masked_text_attention_mask,
            )
            loss = contrastive + self.cmlm_lambda * cmlm
        return VLPOutput(
            loss=loss, image_features=image_features, text_features=text_features,
            logits_per_image=logits_i, logits_per_text=logits_t,
            contrastive_loss=contrastive, cmlm_loss=cmlm,
        )
