#!/usr/bin/env python
"""Read-only preflight checks for the WPE reviewer post-hoc analyses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "yaml",
    "tqdm",
    "torch",
    "transformers",
]

EXPECTED_LENGTHS = {
    "in_sample": {"wb": 13_397_377, "fb": 20_461_603},
    "test1": {"wb": 45_630_757, "fb": 69_651_774},
    "test2": {"wb": 55_287_088, "fb": 92_871_386},
}

PERIOD_DIRS = {
    "in_sample": {"tokenized_subdir": "", "metrics_subdir": "in_sample"},
    "test1": {"tokenized_subdir": "test1", "metrics_subdir": "test1"},
    "test2": {"tokenized_subdir": "test2", "metrics_subdir": "test2"},
}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def check_metric_length(path: Path) -> int | None:
    if not module_available("torch"):
        return None
    import torch

    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu", weights_only=True)
    try:
        return len(obj)
    except TypeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        default="/hdd/shared/AHRQ_LLM/cache",
        help="Directory containing llama3-WPE_EHRlogs-* cache folders.",
    )
    parser.add_argument(
        "--output",
        default="outputs/preflight_environment.json",
        help="Path for the JSON report. This is the only file written.",
    )
    parser.add_argument(
        "--check-lengths",
        action="store_true",
        help="Load metric tensors and compare lengths to handoff totals.",
    )
    args = parser.parse_args()

    cache_root = Path(args.cache_root).resolve()
    wb_root = cache_root / "llama3-WPE_EHRlogs-WB-FBeval-T3"
    fb_root = cache_root / "llama3-WPE_EHRlogs-FB-T3"

    report: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "cache_root": str(cache_root),
        "modules": {name: module_available(name) for name in REQUIRED_MODULES},
        "cache_dirs": {
            "wb": wb_root.is_dir(),
            "fb": fb_root.is_dir(),
        },
        "periods": {},
        "guardrails": [
            "Do not run main_WPE.py.",
            "Do not run markov_baseline_updated.py.",
            "Do not run generate_fixed_split.py.",
            "Write new analysis outputs outside the model cache.",
        ],
    }

    for period, dirs in PERIOD_DIRS.items():
        wb_token_dir = wb_root / dirs["tokenized_subdir"]
        fb_token_dir = fb_root / dirs["tokenized_subdir"]
        wb_metric_dir = wb_root / dirs["metrics_subdir"]
        fb_metric_dir = fb_root / dirs["metrics_subdir"]

        period_report = {
            "wb_tokenized_dataset_test": str(wb_token_dir / "tokenized_dataset_test.pt"),
            "fb_tokenized_dataset_test": str(fb_token_dir / "tokenized_dataset_test.pt"),
            "wb_wpe_ids_candidates": [
                str(p)
                for p in [
                    wb_token_dir / "test_wpe_ids.npy",
                    wb_token_dir / "testset_wpe_ids.npy",
                    wb_root / "testset_wpe_ids.npy",
                ]
                if p.exists()
            ],
            "fb_wpe_ids_candidates": [
                str(p)
                for p in [
                    fb_token_dir / "test_wpe_ids.npy",
                    fb_token_dir / "testset_wpe_ids.npy",
                    fb_root / "testset_wpe_ids.npy",
                ]
                if p.exists()
            ],
            "required_files": {},
            "fallback_files": {},
            "metric_lengths": {},
        }

        required = {
            "wb_tokenized": wb_token_dir / "tokenized_dataset_test.pt",
            "fb_tokenized": fb_token_dir / "tokenized_dataset_test.pt",
            "wb_nn_correct": wb_metric_dir / "llm_tokens_nn_correct.pt",
            "wb_exact_correct": wb_metric_dir / "llm_tokens_correct.pt",
            "fb_correct": fb_metric_dir / "llm_tokens_correct.pt",
            "markov_correct": fb_metric_dir / "markov_tokens_correct.pt",
        }
        fallback = {
            "wb_prompt_nn_correct": wb_metric_dir / "llm-prompt_tokens_nn_correct.pt",
            "wb_prompt_exact_correct": wb_metric_dir / "llm-prompt_tokens_correct.pt",
            "fb_prompt_correct": fb_metric_dir / "llm-prompt_tokens_correct.pt",
        }

        for name, path in required.items():
            period_report["required_files"][name] = {
                "path": str(path),
                "exists": path.exists(),
                "bytes": file_size(path),
            }
        for name, path in fallback.items():
            period_report["fallback_files"][name] = {
                "path": str(path),
                "exists": path.exists(),
                "bytes": file_size(path),
            }

        if args.check_lengths:
            for name, path in {**required, **fallback}.items():
                if name.endswith("correct"):
                    period_report["metric_lengths"][name] = check_metric_length(path)
            expected = EXPECTED_LENGTHS[period]
            period_report["expected_lengths"] = expected
            period_report["length_matches"] = {
                "wb_nn_correct": period_report["metric_lengths"].get("wb_nn_correct")
                == expected["wb"],
                "wb_prompt_nn_correct": period_report["metric_lengths"].get(
                    "wb_prompt_nn_correct"
                )
                == expected["wb"],
                "fb_correct": period_report["metric_lengths"].get("fb_correct")
                == expected["fb"],
                "fb_prompt_correct": period_report["metric_lengths"].get(
                    "fb_prompt_correct"
                )
                == expected["fb"],
                "markov_correct": period_report["metric_lengths"].get("markov_correct")
                == expected["fb"],
            }

        report["periods"][period] = period_report

    missing_modules = [
        name for name, present in report["modules"].items() if not present
    ]
    missing_dirs = [
        name for name, present in report["cache_dirs"].items() if not present
    ]
    report["ok"] = not missing_modules and not missing_dirs
    report["missing_modules"] = missing_modules
    report["missing_cache_dirs"] = missing_dirs

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
