from __future__ import annotations
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, MBartConfig, MBartForConditionalGeneration
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from data.loader import load_language_records
from utils import load_yaml


def collect_training_subtitles(data_config: str, language: str) -> list[str]:
    data_cfg = load_yaml(data_config)
    records, _ = load_language_records(data_cfg, language, split="train")
    return [span.text for record in records for span in record.sentences]


def shrink_mbart_depth(
    model: MBartForConditionalGeneration,
    encoder_layers: int = 3, decoder_layers: int = 3, attention_heads: int = 8,
) -> MBartForConditionalGeneration:
    """Shrink a (vocab-trimmed) mBART to fewer layers, keeping its token embeddings.

    Same construction as GFSLT-VLP's `mytran` step (tools/trim_model.py, arXiv 2307.14768):
    a smaller mBART is built RANDOMLY INITIALIZED from the shrunk config and only the trimmed
    pretrained token embeddings are copied in (`mytran_model.model.shared = trimmed.model.shared`).
    The deep pretrained transformer layers are NOT inherited — they are pretrained from scratch in
    stage-1 VLP. We keep mBART's default tie_word_embeddings=True so the copied embeddings double as
    the output projection (mytran's config left lm_head random by setting it false; tying is cleaner
    and parameter-free).

    Unlike GFSLT-VLP — which kept the FULL trimmed mBART as the stage-1 text encoder and used the
    small model only on the visual side — this single shrunk model is used everywhere (text encoder
    in stage-1 VLP, encoder + AR/DLM decoder downstream). Deliberate speed-motivated simplification.
    """
    config = MBartConfig.from_dict(model.config.to_dict())
    config.encoder_layers = int(encoder_layers)
    config.decoder_layers = int(decoder_layers)
    config.num_hidden_layers = int(encoder_layers)
    config.encoder_attention_heads = int(attention_heads)
    config.decoder_attention_heads = int(attention_heads)
    config.tie_word_embeddings = True

    small = MBartForConditionalGeneration(config)
    with torch.no_grad():
        small.model.shared.weight.copy_(model.model.shared.weight)
        small.model.encoder.embed_tokens.weight.copy_(model.model.shared.weight)
        small.model.decoder.embed_tokens.weight.copy_(model.model.shared.weight)
    small.tie_weights()
    return small


def trim_MBartForConditionalGeneration(
    data_config: str, language: str, mbart_name: str,
    target_lang: str, tokenizer_out: str, model_out: str,
    encoder_layers: int | None = None, decoder_layers: int | None = None, attention_heads: int = 8,
) -> dict[str, str | int]:
    """Trim mBART's vocabulary to the training subtitles, then (optionally) its depth.

    Writes 1 directory (`tokenizer_out` == `model_out` by convention) holding the trimmed  tokenizer and the final model. 
    When `encoder_layers`/`decoder_layers` are given, the vocab-trimmed model is shrunk in depth before saving, so the saved 
    model is the small one used by every stage — there is no second model dir.
    """
    subtitles = collect_training_subtitles(data_config, language)
    if not subtitles: raise RuntimeError(f"No train subtitles found for language={language}")
    tokenizer = AutoTokenizer.from_pretrained(mbart_name, src_lang=target_lang, tgt_lang=target_lang, use_fast=False)
    tokenizer_trimmer = TokenizerTrimmer(tokenizer)
    tokenizer_trimmer.make_vocab(subtitles)
    tokenizer_trimmer.make_tokenizer()
    Path(tokenizer_out).mkdir(parents=True, exist_ok=True)
    tokenizer_trimmer.trimmed_tokenizer.save_pretrained(tokenizer_out)

    model = MBartForConditionalGeneration.from_pretrained(mbart_name)
    model_trimmer = MBartTrimmer(model, model.config, tokenizer_trimmer.trimmed_tokenizer)
    model_trimmer.make_weights(tokenizer_trimmer.trimmed_vocab_ids)
    model_trimmer.make_model()
    final_model = model_trimmer.trimmed_model

    result: dict[str, str | int] = {
        "language": language, "subtitles": len(subtitles),
        "trimmed_vocab_ids": len(tokenizer_trimmer.trimmed_vocab_ids),
        "tokenizer": tokenizer_out, "model": model_out,
        "encoder_layers": int(final_model.config.encoder_layers),
        "decoder_layers": int(final_model.config.decoder_layers),
    }
    if encoder_layers is not None or decoder_layers is not None:
        final_model = shrink_mbart_depth(
            final_model,
            encoder_layers=int(encoder_layers if encoder_layers is not None else final_model.config.encoder_layers),
            decoder_layers=int(decoder_layers if decoder_layers is not None else final_model.config.decoder_layers),
            attention_heads=int(attention_heads),
        )
        result["encoder_layers"] = int(final_model.config.encoder_layers)
        result["decoder_layers"] = int(final_model.config.decoder_layers)

    Path(model_out).mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(model_out)
    result["parameters"] = sum(p.numel() for p in final_model.parameters())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim mBART tokenizer/model to local YouTube-SL subtitles")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--language", default="asf")
    parser.add_argument("--mbart-name", default="facebook/mbart-large-cc25")
    parser.add_argument("--target-lang", default="en_XX")
    parser.add_argument("--tokenizer-out", default="checkpoints/trimmed_mbart_asf")
    parser.add_argument("--model-out", default="checkpoints/trimmed_mbart_asf")
    parser.add_argument("--encoder-layers", type=int, default=None, help="Shrink to N encoder layers (omit to keep full depth)")
    parser.add_argument("--decoder-layers", type=int, default=None, help="Shrink to N decoder layers (omit to keep full depth)")
    parser.add_argument("--attention-heads", type=int, default=8)
    args = parser.parse_args()

    result = trim_MBartForConditionalGeneration(
        data_config=args.data_config, language=args.language, mbart_name=args.mbart_name,
        target_lang=args.target_lang, tokenizer_out=args.tokenizer_out, model_out=args.model_out,
        encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, attention_heads=args.attention_heads,
    )
    print(f"Collected {result['subtitles']} subtitles")
    print(f"Trimmed vocab ids: {result['trimmed_vocab_ids']}")
    print(f"Saved tokenizer to {result['tokenizer']}")
    print(f"Saved mBART ({result['parameters'] / 1e6:.1f}M params, "
          f"{result['encoder_layers']}enc/{result['decoder_layers']}dec) to {result['model']}")