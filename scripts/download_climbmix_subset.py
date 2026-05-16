#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_TRAIN_TOKENS = 8 * 1024 * 6104
DEFAULT_VALIDATION_TOKENS = 2 * 8 * (1024 + 1) * 128


def _write_split(
    samples: Iterable[dict[str, Any]],
    *,
    output_path: Path,
    token_budget: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_written = 0
    rows_written = 0

    with output_path.open("w") as f:
        for sample in samples:
            tokens = list(sample["tokens"])
            if not tokens:
                continue

            remaining = token_budget - tokens_written
            if remaining <= 0:
                break
            if len(tokens) > remaining:
                tokens = tokens[:remaining]

            f.write(
                json.dumps(
                    {
                        "tokens": tokens,
                        "token_count": len(tokens),
                        "cluster_id": sample.get("cluster_id"),
                    }
                )
                + "\n"
            )
            tokens_written += len(tokens)
            rows_written += 1

            if tokens_written >= token_budget:
                break

    if tokens_written < token_budget:
        raise RuntimeError(
            f"Only wrote {tokens_written:,} tokens to {output_path}; "
            f"requested {token_budget:,}."
        )

    return tokens_written, rows_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a deterministic local subset of pre-tokenized "
            "nvidia/Nemotron-ClimbMix."
        )
    )
    parser.add_argument(
        "--dataset",
        default="nvidia/Nemotron-ClimbMix",
        help="Hugging Face dataset repo to stream from.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/climbmix_50m"),
        help="Directory to write train.jsonl, validation.jsonl, and metadata.json.",
    )
    parser.add_argument(
        "--train-tokens",
        type=int,
        default=DEFAULT_TRAIN_TOKENS,
        help="Number of GPT-2 tokens to write to train.jsonl.",
    )
    parser.add_argument(
        "--validation-tokens",
        type=int,
        default=DEFAULT_VALIDATION_TOKENS,
        help="Number of GPT-2 tokens to write to validation.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory.",
    )
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output} already exists. Pass --overwrite to replace it."
        )
    args.output.mkdir(parents=True, exist_ok=True)

    train_path = args.output / "train.jsonl"
    validation_path = args.output / "validation.jsonl"
    metadata_path = args.output / "metadata.json"
    if args.overwrite:
        for path in (train_path, validation_path, metadata_path):
            path.unlink(missing_ok=True)

    samples = iter(
        load_dataset(args.dataset, name="default", split="train", streaming=True)
    )
    train_tokens, train_rows = _write_split(
        samples,
        output_path=train_path,
        token_budget=args.train_tokens,
    )
    validation_tokens, validation_rows = _write_split(
        samples,
        output_path=validation_path,
        token_budget=args.validation_tokens,
    )

    metadata = {
        "source_dataset": args.dataset,
        "tokenizer": "gpt2",
        "train": {
            "path": train_path.name,
            "tokens": train_tokens,
            "rows": train_rows,
        },
        "validation": {
            "path": validation_path.name,
            "tokens": validation_tokens,
            "rows": validation_rows,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {train_tokens:,} train tokens to {train_path}")
    print(f"Wrote {validation_tokens:,} validation tokens to {validation_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
