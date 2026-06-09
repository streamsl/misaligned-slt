from __future__ import annotations
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class MBartARDecoder(nn.Module): # Autoregressive mBART decoder wrapper for the fair AR-vs-DLM comparison
    def __init__(self, mbart: MBartForConditionalGeneration, forced_bos_token_id: int | None = None):
        super().__init__()
        self.mbart = mbart
        self.forced_bos_token_id = forced_bos_token_id

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, forced_bos_token_id: int | None = None) -> "MBartARDecoder":
        return cls(MBartForConditionalGeneration.from_pretrained(model_name_or_path), forced_bos_token_id)

    def forward(
        self, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None, decoder_input_ids: torch.Tensor | None = None,
    ):
        return self.mbart(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask,
            labels=labels, decoder_input_ids=decoder_input_ids, return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor,
        max_new_tokens: int = 128, **kwargs,
    ) -> torch.Tensor:
        generate_kwargs = dict(kwargs)
        if self.forced_bos_token_id is not None: generate_kwargs.setdefault("forced_bos_token_id", self.forced_bos_token_id)
        return self.mbart.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=encoder_attention_mask, max_new_tokens=max_new_tokens, **generate_kwargs,
        )
