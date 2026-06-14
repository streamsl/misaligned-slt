from __future__ import annotations
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, MBartConfig, MBartForConditionalGeneration
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from data.loader import load_language_records
from utils import load_yaml, mbart_name, mbart_trimmed_dir


def collect_training_subtitles(data_config: str, language: str) -> list[str]:
    data_cfg = load_yaml(data_config)
    records, _ = load_language_records(data_cfg, language, split="train")
    return [span.text for record in records for span in record.sentences]


def shrink_mbart_depth(
    model: MBartForConditionalGeneration,
    encoder_layers: int = 3, decoder_layers: int = 3, attention_heads: int | None = None,
) -> MBartForConditionalGeneration:
    """Shrink a (vocab-trimmed) mBART to fewer layers by TRUNCATION-INIT.

    Builds a shallow mBART and copies every shape-matching pretrained tensor from `model`: the token
    embeddings, positional embeddings, embedding/final layer norms, and the FIRST `encoder_layers` /
    `decoder_layers` transformer blocks. Surplus deep layers are dropped; nothing is left random.

    Why not GFSLT-VLP's `mytran` (random-init layers + embeddings only): mytran was random because
    GFSLT-VLP paired it with a SEPARATE full pretrained mBART as the stage-1 text encoder, which gave
    the contrastive loss a stable target. We use ONE small model everywhere (text encoder, visual
    encoder, AR/DLM decoder), so a random-init text encoder leaves the contrastive/CMLM objectives
    with no pretrained text space to align to and stage-1 VLP val-loss diverges. Truncation-init keeps
    the model small and fast (layer count is the only size lever at fixed d_model) while retaining
    pretrained language structure — the standard shallow-from-deep init ("Well-Read Students Learn
    Better", arXiv 1908.08962). `attention_heads=None` keeps the source head count so the copied
    attention projections transfer exactly (head count does not change parameter count at fixed d_model).
    """
    config = MBartConfig.from_dict(model.config.to_dict())
    config.encoder_layers = int(encoder_layers)
    config.decoder_layers = int(decoder_layers)
    config.num_hidden_layers = int(encoder_layers)
    if attention_heads is not None:
        config.encoder_attention_heads = int(attention_heads)
        config.decoder_attention_heads = int(attention_heads)
    config.tie_word_embeddings = True

    small = MBartForConditionalGeneration(config)
    # Copy every tensor whose name + shape matches: embeddings, layer norms, positional embeddings, and
    # encoder/decoder layers 0..N-1 (the small model simply has no keys for the dropped deeper layers).
    small_sd = small.state_dict()
    copied = 0
    for name, tensor in model.state_dict().items():
        if name in small_sd and small_sd[name].shape == tensor.shape:
            small_sd[name] = tensor.clone()
            copied += 1
    small.load_state_dict(small_sd)
    small.tie_weights()
    small._trim_copied_tensors = copied  # surfaced by trim_MBart for the run log
    return small


def trim_MBartForConditionalGeneration(
    data_config: str, language: str, mbart_name: str,
    target_lang: str, tokenizer_out: str, model_out: str,
    encoder_layers: int | None = None, decoder_layers: int | None = None, attention_heads: int | None = None,
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
            attention_heads=None if attention_heads is None else int(attention_heads),
        )
        result["encoder_layers"] = int(final_model.config.encoder_layers)
        result["decoder_layers"] = int(final_model.config.decoder_layers)
        result["truncation_copied_tensors"] = int(getattr(final_model, "_trim_copied_tensors", 0))

    Path(model_out).mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(model_out)
    result["parameters"] = sum(p.numel() for p in final_model.parameters())
    return result


if __name__ == "__main__":
    # Prefer `python train.py --stage trim-mbart` (same logic). This standalone CLI resolves every
    # default from the configs too, so a bare `python trim_mbart.py` is correct for the active dataset
    # (e.g. phoenix → target_lang de_DE, depth-trim from mbart.layers) — no stale en_XX/asf defaults.
    parser = argparse.ArgumentParser(description="Trim mBART tokenizer/model to the active dataset's training subtitles")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--stage1-config", default="configs/stage1_vlp.yaml")
    parser.add_argument("--language", default=None, help="Default: stage1-config language / data.yaml active_languages[0]")
    parser.add_argument("--mbart-name", default=None, help="Default: mbart.name from the stage1 config")
    parser.add_argument("--target-lang", default=None, help="Default: the language entry's target_lang in data.yaml (e.g. de_DE for phoenix)")
    parser.add_argument("--tokenizer-out", default=None, help="Default: mbart.trimmed_dir from the stage1 config")
    parser.add_argument("--model-out", default=None, help="Default: mbart.trimmed_dir from the stage1 config")
    parser.add_argument("--encoder-layers", type=int, default=None, help="Truncate to first N encoder layers (default: mbart.layers.encoder)")
    parser.add_argument("--decoder-layers", type=int, default=None, help="Truncate to first N decoder layers (default: mbart.layers.decoder)")
    parser.add_argument(
        "--attention-heads", type=int, default=None,
        help="Override head count (default: mbart.layers.attention_heads, else the source's)"
    )
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    stage1_cfg = load_yaml(args.stage1_config)
    language = str(args.language or stage1_cfg.get("language", data_cfg.get("active_languages", ["phoenix"])[0]))
    target_lang = args.target_lang or data_cfg["languages"][language].get("target_lang", "en_XX")
    trimmed_dir = mbart_trimmed_dir(stage1_cfg)
    layers_cfg = stage1_cfg.get("mbart", {}).get("layers", {})
    result = trim_MBartForConditionalGeneration(
        data_config=args.data_config, language=language,
        mbart_name=args.mbart_name or mbart_name(stage1_cfg), target_lang=target_lang,
        tokenizer_out=args.tokenizer_out or trimmed_dir, model_out=args.model_out or trimmed_dir,
        encoder_layers=args.encoder_layers if args.encoder_layers is not None else layers_cfg.get("encoder"),
        decoder_layers=args.decoder_layers if args.decoder_layers is not None else layers_cfg.get("decoder"),
        attention_heads=args.attention_heads if args.attention_heads is not None else layers_cfg.get("attention_heads"),
    )
    print(f"Collected {result['subtitles']} subtitles")
    print(f"Trimmed vocab ids: {result['trimmed_vocab_ids']}")
    print(f"Saved tokenizer to {result['tokenizer']}")
    print(f"Saved mBART ({result['parameters'] / 1e6:.1f}M params, "
          f"{result['encoder_layers']}enc/{result['decoder_layers']}dec) to {result['model']}")