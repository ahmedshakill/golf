#!/usr/bin/env python3
"""
Prepare local FineWeb train/val files for HERMES.

Defaults to plain UTF-8 text files because that is the safest local path.
Optionally writes uint16 token bins using MicroBPE from train_hermes.py.
"""

import argparse
import os
from array import array

from datasets import load_dataset

from train_hermes import MicroBPE


def resolve_hf_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_TOKEN")


def iter_text_samples(subset: str, hf_token: str | None = None):
    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=subset,
        split="train",
        streaming=True,
        token=hf_token,
    )
    for sample in dataset:
        text = sample.get("text", "").strip()
        if text:
            yield text


def write_text_split(samples, out_path: str, target_bytes: int) -> tuple[int, int]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written_docs = 0
    written_bytes = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for text in samples:
            payload = text + "\n"
            encoded = payload.encode("utf-8")
            f.write(payload)
            written_docs += 1
            written_bytes += len(encoded)
            if written_bytes >= target_bytes:
                break

    return written_docs, written_bytes


def write_bin_split(samples, out_path: str, target_bytes: int, tokenizer: MicroBPE) -> tuple[int, int]:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    written_docs = 0
    written_bytes = 0
    token_buffer = array("H")
    flush_every = 1 << 20

    with open(out_path, "wb") as f:
        for text in samples:
            token_buffer.extend(tokenizer.encode(text))
            written_docs += 1
            if len(token_buffer) >= flush_every:
                token_buffer.tofile(f)
                written_bytes += len(token_buffer) * 2
                token_buffer = array("H")
                if written_bytes >= target_bytes:
                    break

        if len(token_buffer) > 0 and written_bytes < target_bytes:
            remaining_tokens = max(0, (target_bytes - written_bytes) // 2)
            if remaining_tokens:
                token_buffer = array("H", token_buffer[:remaining_tokens])
            token_buffer.tofile(f)
            written_bytes += len(token_buffer) * 2

    return written_docs, written_bytes


def default_outputs(fmt: str) -> tuple[str, str]:
    suffix = ".bin" if fmt == "bin" else ""
    return f"data/fineweb_train{suffix}", f"data/fineweb_val{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Prepare a local FineWeb slice")
    parser.add_argument("--subset", default="CC-MAIN-2024-10",
                        help="FineWeb subset/config, e.g. CC-MAIN-2024-10")
    parser.add_argument("--format", choices=["text", "bin"], default="text",
                        help="Output plain text files or uint16 token bins")
    parser.add_argument("--train-out", default=None, help="Output path for training split")
    parser.add_argument("--val-out", default=None, help="Output path for validation split")
    parser.add_argument("--train-mb", type=int, default=64,
                        help="Approximate train split size in MiB")
    parser.add_argument("--val-mb", type=int, default=8,
                        help="Approximate validation split size in MiB")
    parser.add_argument("--hf_token", default=None,
                        help="Optional Hugging Face token. Falls back to HF_TOKEN or HUGGINGFACE_HUB_TOKEN env vars.")
    args = parser.parse_args()

    default_train, default_val = default_outputs(args.format)
    train_out = args.train_out or default_train
    val_out = args.val_out or default_val

    hf_token = resolve_hf_token(args.hf_token)

    print(f"[FineWeb] Streaming subset={args.subset} format={args.format}")
    samples = iter_text_samples(args.subset, hf_token=hf_token)

    if args.format == "text":
        train_docs, train_bytes = write_text_split(samples, train_out, args.train_mb * 1024 * 1024)
        print(f"[FineWeb] Wrote train: {train_out} | docs={train_docs} | bytes={train_bytes}")
        val_docs, val_bytes = write_text_split(samples, val_out, args.val_mb * 1024 * 1024)
        print(f"[FineWeb] Wrote val:   {val_out} | docs={val_docs} | bytes={val_bytes}")
    else:
        tokenizer = MicroBPE()
        train_docs, train_bytes = write_bin_split(samples, train_out, args.train_mb * 1024 * 1024, tokenizer)
        print(f"[FineWeb] Wrote train: {train_out} | docs={train_docs} | bytes={train_bytes}")
        val_docs, val_bytes = write_bin_split(samples, val_out, args.val_mb * 1024 * 1024, tokenizer)
        print(f"[FineWeb] Wrote val:   {val_out} | docs={val_docs} | bytes={val_bytes}")


if __name__ == "__main__":
    main()
