from __future__ import annotations
import argparse
from pathlib import Path
from transformers import AutoTokenizer, MBartForConditionalGeneration
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer
from data.loader import load_language_records
from utils import load_yaml


def collect_training_subtitles(data_config: str, language: str) -> list[str]:
    data_cfg = load_yaml(data_config)
    records, _ = load_language_records(data_cfg, language, split="train")
    return [span.text for record in records for span in record.sentences]


def trim_MBartForConditionalGeneration(
    data_config: str, language: str, mbart_name: str,
    target_lang: str, tokenizer_out: str, model_out: str,
) -> dict[str, str | int]:
    subtitles = collect_training_subtitles(data_config, language)
    if not subtitles: raise RuntimeError(f"No train subtitles found for language={language}")
    tokenizer = AutoTokenizer.from_pretrained(=mbart_name, src_lang=target_lang, tgt_lang=target_lang, use_fast=False)
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
    parser.add_argument("--tokenizer-out", default="captioners/trimmed_mbart_asf")
    parser.add_argument("--model-out", default="captioners/trimmed_mbart_asf")
    args = parser.parse_args()

    result = trim_MBartForConditionalGeneration(
        data_config=args.data_config, language=args.language, mbart_name=args.mbart_name,
        target_lang=args.target_lang, tokenizer_out=args.tokenizer_out, model_out=args.model_out,
    )
    print(f"Collected {result['subtitles']} subtitles")
    print(f"Trimmed vocab ids: {result['trimmed_vocab_ids']}")
    print(f"Saved tokenizer to {result['tokenizer']}")
    print(f"Saved mBART to {result['model']}")