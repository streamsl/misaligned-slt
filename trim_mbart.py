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


def build_small_visual_mbart(
    trimmed_dir: str, out_dir: str,
    encoder_layers: int = 3, decoder_layers: int = 3, attention_heads: int = 8,
    activation_function: str = "relu",
) -> dict[str, str | int]:
    """Build the GFSLT-VLP "mytran"-style small visual-side mBART.

    Protocol from GFSLT-VLP tools/trim_model.py + pretrain_models/mytran/config.json
    (arXiv 2307.14768): a small mBART (3 encoder / 3 decoder layers, 8 heads, d_model 1024,
    ffn 4096, relu) is built RANDOMLY INITIALIZED, and only the trimmed pretrained token
    embeddings are copied in (`mytran_model.model.shared = new_model.model.shared`). The
    pretrained 12-layer weights are NOT inherited — stage-1 VLP pretrains this model.

    Divergence from mytran's config (documented): we keep mBART's default
    tie_word_embeddings=True (mytran sets false, leaving lm_head random). Tying makes the
    copied embeddings double as the pretrained output projection at zero extra parameters.
    """
    trimmed = MBartForConditionalGeneration.from_pretrained(trimmed_dir)
    config = MBartConfig.from_pretrained(trimmed_dir)
    config.encoder_layers = int(encoder_layers)
    config.decoder_layers = int(decoder_layers)
    config.num_hidden_layers = int(encoder_layers)
    config.encoder_attention_heads = int(attention_heads)
    config.decoder_attention_heads = int(attention_heads)
    config.activation_function = str(activation_function)  # mytran config.json: relu
    config.attention_dropout = 0.0                          # mytran config.json
    config.activation_dropout = 0.0                         # mytran config.json
    config.dropout = 0.1                                    # mytran config.json
    config.tie_word_embeddings = True

    small = MBartForConditionalGeneration(config)
    with torch.no_grad():
        small.model.shared.weight.copy_(trimmed.model.shared.weight)
        small.model.encoder.embed_tokens.weight.copy_(trimmed.model.shared.weight)
        small.model.decoder.embed_tokens.weight.copy_(trimmed.model.shared.weight)
        if small.lm_head.weight.data_ptr() != small.model.shared.weight.data_ptr():
            small.lm_head.weight.copy_(trimmed.lm_head.weight)
    small.tie_weights()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    small.save_pretrained(out_dir)
    n_params = sum(p.numel() for p in small.parameters())
    return {"visual_mbart": out_dir, "encoder_layers": int(encoder_layers),
            "decoder_layers": int(decoder_layers), "parameters": n_params}


def trim_MBartForConditionalGeneration(
    data_config: str, language: str, mbart_name: str,
    target_lang: str, tokenizer_out: str, model_out: str,
) -> dict[str, str | int]:
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
    Path(model_out).mkdir(parents=True, exist_ok=True)
    model_trimmer.trimmed_model.save_pretrained(model_out)
    return {
        "language": language, "subtitles": len(subtitles),
        "trimmed_vocab_ids": len(tokenizer_trimmer.trimmed_vocab_ids),
        "tokenizer": tokenizer_out, "model": model_out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim mBART tokenizer/model to local YouTube-SL subtitles")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--language", default="asf")
    parser.add_argument("--mbart-name", default="facebook/mbart-large-cc25")
    parser.add_argument("--target-lang", default="en_XX")
    parser.add_argument("--tokenizer-out", default="checkpoints/trimmed_mbart_asf")
    parser.add_argument("--model-out", default="checkpoints/trimmed_mbart_asf")
    parser.add_argument("--small-out", default=None, help="Also build the GFSLT-VLP mytran-style small visual mBART here")
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=8)
    args = parser.parse_args()

    result = trim_MBartForConditionalGeneration(
        data_config=args.data_config, language=args.language, mbart_name=args.mbart_name,
        target_lang=args.target_lang, tokenizer_out=args.tokenizer_out, model_out=args.model_out,
    )
    print(f"Collected {result['subtitles']} subtitles")
    print(f"Trimmed vocab ids: {result['trimmed_vocab_ids']}")
    print(f"Saved tokenizer to {result['tokenizer']}")
    print(f"Saved mBART to {result['model']}")
    if args.small_out:
        small = build_small_visual_mbart(
            trimmed_dir=args.model_out, out_dir=args.small_out,
            encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers,
            attention_heads=args.attention_heads,
        )
        print(f"Saved small visual mBART ({small['parameters'] / 1e6:.1f}M params) to {small['visual_mbart']}")