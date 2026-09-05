#!/usr/bin/env python
"""Read-only post-hoc analyses for WPE FB/WB matched action evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


EXPECTED_LENGTHS = {
    "in_sample": {"wb": 13_397_377, "fb": 20_461_603},
    "test1": {"wb": 45_630_757, "fb": 69_651_774},
    "test2": {"wb": 55_287_088, "fb": 92_871_386},
}

PERIOD_LAYOUT = {
    "in_sample": {"tokenized_subdir": "", "metrics_subdir": "in_sample"},
    "test1": {"tokenized_subdir": "test1", "metrics_subdir": "test1"},
    "test2": {"tokenized_subdir": "test2", "metrics_subdir": "test2"},
}


@dataclass(frozen=True)
class AnalysisPaths:
    cache_root: Path
    wb_root: Path
    fb_root: Path
    period: str
    wb_tokenized: Path
    fb_tokenized: Path
    wb_metric_dir: Path
    fb_metric_dir: Path
    wb_tokenizer_dir: Path
    fb_tokenizer_dir: Path
    wb_wpe_ids: Path
    fb_wpe_ids: Path


def build_paths(args: argparse.Namespace) -> AnalysisPaths:
    cache_root = args.cache_root.resolve()
    wb_root = cache_root / args.wb_model_dir
    fb_root = cache_root / args.fb_model_dir
    layout = PERIOD_LAYOUT[args.period]
    wb_tokenized_subdir = layout["tokenized_subdir"] if args.wb_tokenized_subdir is None else args.wb_tokenized_subdir
    fb_tokenized_subdir = layout["tokenized_subdir"] if args.fb_tokenized_subdir is None else args.fb_tokenized_subdir
    wb_metrics_subdir = layout["metrics_subdir"] if args.wb_metrics_subdir is None else args.wb_metrics_subdir
    fb_metrics_subdir = layout["metrics_subdir"] if args.fb_metrics_subdir is None else args.fb_metrics_subdir

    wb_token_dir = wb_root / wb_tokenized_subdir
    fb_token_dir = fb_root / fb_tokenized_subdir
    wb_metric_dir = wb_root / wb_metrics_subdir
    fb_metric_dir = fb_root / fb_metrics_subdir

    return AnalysisPaths(
        cache_root=cache_root,
        wb_root=wb_root,
        fb_root=fb_root,
        period=args.period,
        wb_tokenized=wb_token_dir / "tokenized_dataset_test.pt",
        fb_tokenized=fb_token_dir / "tokenized_dataset_test.pt",
        wb_metric_dir=wb_metric_dir,
        fb_metric_dir=fb_metric_dir,
        wb_tokenizer_dir=resolve_tokenizer_dir(wb_root),
        fb_tokenizer_dir=resolve_tokenizer_dir(fb_root),
        wb_wpe_ids=resolve_wpe_ids(wb_root, wb_token_dir),
        fb_wpe_ids=resolve_wpe_ids(fb_root, fb_token_dir),
    )


def resolve_tokenizer_dir(model_root: Path) -> Path:
    candidates = [
        model_root / "best_eval_loss_checkpoint",
        model_root,
    ]
    for path in candidates:
        if (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists():
            return path
    raise FileNotFoundError(f"No tokenizer files found in candidates: {candidates}")


def resolve_wpe_ids(model_root: Path, token_dir: Path) -> Path:
    candidates = [
        token_dir / "test_wpe_ids.npy",
        token_dir / "testset_wpe_ids.npy",
        model_root / "testset_wpe_ids.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No WPE ID file found in candidates: {candidates}")


def metric_path(metric_dir: Path, prefix: str, metric: str) -> Path:
    return metric_dir / f"{prefix}_tokens_{metric}.pt"


def load_tensor_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return np.asarray(obj)


def load_tokenized(path: Path) -> list[dict[str, torch.Tensor]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, list):
        raise TypeError(f"Expected list from {path}, got {type(data)}")
    return data


def valid_token_ids(tokenizer: AutoTokenizer, tokens: Iterable[str]) -> list[int]:
    ids = tokenizer.convert_tokens_to_ids(list(tokens))
    if not isinstance(ids, list):
        ids = [ids]
    return [int(token_id) for token_id in ids if token_id is not None]


def count_wb_chunks(
    tokenized: list[dict[str, torch.Tensor]],
    tokenizer: AutoTokenizer,
    skip_n_tokens: int,
) -> np.ndarray:
    row_token_id = tokenizer.convert_tokens_to_ids("<ROW>")
    if row_token_id is None:
        raise ValueError("WB tokenizer does not define <ROW>; cannot count WB chunks.")
    pad_token_id = tokenizer.pad_token_id
    counts = np.empty(len(tokenized), dtype=np.int64)

    for idx, item in enumerate(tokenized):
        input_ids = item["input_ids"]
        labels = torch.roll(input_ids, shifts=-1, dims=0)
        labels[-1] = -100
        labels = labels[skip_n_tokens:]
        mask = labels != -100
        if pad_token_id is not None:
            mask &= labels != pad_token_id
        labels_masked = labels[mask]

        row_positions = torch.nonzero(labels_masked == row_token_id, as_tuple=False).flatten()
        if len(row_positions) == 0:
            counts[idx] = 0
            continue
        first_nonempty = int(row_positions[0].item() > 0)
        later_nonempty = int((torch.diff(row_positions) > 1).sum().item()) if len(row_positions) > 1 else 0
        counts[idx] = first_nonempty + later_nonempty

    return counts


def count_fb_actions(
    tokenized: list[dict[str, torch.Tensor]],
    tokenizer: AutoTokenizer,
    skip_n_tokens: int,
) -> np.ndarray:
    exclude_ids = []
    if tokenizer.pad_token_id is not None:
        exclude_ids.append(int(tokenizer.pad_token_id))
    exclude_ids.extend(valid_token_ids(tokenizer, ["<ROW>", "<FIRST_ROW>"]))
    exclude_ids.extend(valid_token_ids(tokenizer, ["[TD_0]", "[TD_10]", "[TD_60]", "[TD_>60]"]))
    exclude_ids = sorted(set(exclude_ids))

    counts = np.empty(len(tokenized), dtype=np.int64)
    for idx, item in enumerate(tokenized):
        labels = item["labels"][skip_n_tokens:]
        mask = labels != -100
        for token_id in exclude_ids:
            mask &= labels != token_id
        counts[idx] = int(mask.sum().item())
    return counts


def split_sums(flat: np.ndarray, counts: np.ndarray) -> np.ndarray:
    if int(counts.sum()) != len(flat):
        raise ValueError(f"Counts sum {counts.sum()} does not match vector length {len(flat)}")
    sums = np.empty(len(counts), dtype=np.float64)
    cursor = 0
    for idx, count in enumerate(counts):
        next_cursor = cursor + int(count)
        sums[idx] = float(np.asarray(flat[cursor:next_cursor], dtype=np.float64).sum())
        cursor = next_cursor
    return sums


def restrict_last_k_sums(flat: np.ndarray, source_counts: np.ndarray, keep_counts: np.ndarray) -> np.ndarray:
    if len(source_counts) != len(keep_counts):
        raise ValueError("source_counts and keep_counts must have the same length")
    if int(source_counts.sum()) != len(flat):
        raise ValueError(f"Source counts sum {source_counts.sum()} does not match vector length {len(flat)}")
    too_large = np.where(keep_counts > source_counts)[0]
    if len(too_large):
        example = int(too_large[0])
        raise ValueError(
            f"Cannot keep K_i larger than source count; first failure sequence {example}: "
            f"K={keep_counts[example]}, source={source_counts[example]}"
        )

    sums = np.empty(len(source_counts), dtype=np.float64)
    cursor = 0
    for idx, count in enumerate(source_counts):
        count = int(count)
        keep = int(keep_counts[idx])
        seg = flat[cursor : cursor + count]
        sums[idx] = float(np.asarray(seg[count - keep : count], dtype=np.float64).sum()) if keep else 0.0
        cursor += count
    return sums


def pooled_mean(seq_sums: np.ndarray, seq_counts: np.ndarray) -> float:
    denom = float(seq_counts.sum())
    if denom == 0:
        return float("nan")
    return float(seq_sums.sum() / denom)


def cluster_bootstrap_ci(
    seq_sums: np.ndarray,
    seq_counts: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_clusters = len(seq_sums)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        stats[i] = seq_sums[idx].sum() / seq_counts[idx].sum()
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(stats.mean()), float(low), float(high)


def ensure_counts(paths: AnalysisPaths, args: argparse.Namespace, out_dir: Path) -> Path:
    counts_path = out_dir / f"{paths.period}_sequence_counts.npz"
    if counts_path.exists() and not args.force_counts:
        return counts_path

    wb_tokenizer = AutoTokenizer.from_pretrained(paths.wb_tokenizer_dir)
    fb_tokenizer = AutoTokenizer.from_pretrained(paths.fb_tokenizer_dir)

    wb_data = load_tokenized(paths.wb_tokenized)
    fb_data = load_tokenized(paths.fb_tokenized)
    if len(wb_data) != len(fb_data):
        raise ValueError(f"WB and FB sequence counts differ: {len(wb_data)} vs {len(fb_data)}")

    wb_wpe_ids = np.load(paths.wb_wpe_ids, allow_pickle=True)
    fb_wpe_ids = np.load(paths.fb_wpe_ids, allow_pickle=True)
    if not np.array_equal(wb_wpe_ids, fb_wpe_ids):
        raise ValueError(f"WB and FB WPE IDs are not aligned: {paths.wb_wpe_ids} vs {paths.fb_wpe_ids}")

    wb_counts = count_wb_chunks(wb_data, wb_tokenizer, args.skip_n_tokens)
    fb_counts = count_fb_actions(fb_data, fb_tokenizer, args.skip_n_tokens)

    np.savez_compressed(
        counts_path,
        wb_counts=wb_counts,
        fb_counts=fb_counts,
        wb_wpe_ids=wb_wpe_ids,
        fb_wpe_ids=fb_wpe_ids,
        skip_n_tokens=np.array(args.skip_n_tokens),
        period=np.array(paths.period),
        wb_tokenized=np.array(str(paths.wb_tokenized)),
        fb_tokenized=np.array(str(paths.fb_tokenized)),
    )
    return counts_path


def analysis1(paths: AnalysisPaths, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    exact = load_tensor_array(metric_path(paths.wb_metric_dir, args.wb_prefix, "correct"))
    nn = load_tensor_array(metric_path(paths.wb_metric_dir, args.wb_prefix, "nn_correct"))
    if len(exact) != len(nn):
        raise ValueError(f"WB exact and NN vector lengths differ: {len(exact)} vs {len(nn)}")

    df = pd.DataFrame(
        [
            {
                "period": paths.period,
                "model": "WB",
                "metric": "exact_match_accuracy_pre_mapping",
                "n_actions": len(exact),
                "value": float(np.mean(exact)),
            },
            {
                "period": paths.period,
                "model": "WB",
                "metric": "semantic_nn_mapped_accuracy",
                "n_actions": len(nn),
                "value": float(np.mean(nn)),
            },
            {
                "period": paths.period,
                "model": "WB",
                "metric": "nn_mapping_rescued_accuracy",
                "n_actions": len(nn),
                "value": float(np.mean((exact == 0) & (nn == 1))),
            },
        ]
    )
    df.to_csv(out_dir / f"{paths.period}_analysis1_wb_mapping.csv", index=False)
    return df


def analysis2(paths: AnalysisPaths, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    counts_path = ensure_counts(paths, args, out_dir)
    counts = np.load(counts_path, allow_pickle=True)
    wb_counts = counts["wb_counts"]
    fb_counts = counts["fb_counts"]

    wb_nn = load_tensor_array(metric_path(paths.wb_metric_dir, args.wb_prefix, "nn_correct"))
    wb_exact = load_tensor_array(metric_path(paths.wb_metric_dir, args.wb_prefix, "correct"))
    markov_correct = load_tensor_array(paths.fb_metric_dir / "markov_tokens_correct.pt")
    fb_correct = None
    if args.comparison == "full":
        fb_correct = load_tensor_array(metric_path(paths.fb_metric_dir, args.fb_prefix, "correct"))

    expected = None if args.skip_expected_lengths else EXPECTED_LENGTHS.get(paths.period)
    validations = {
        "sum_wb_counts": int(wb_counts.sum()),
        "sum_fb_counts": int(fb_counts.sum()),
        "len_wb_nn": int(len(wb_nn)),
        "len_wb_exact": int(len(wb_exact)),
        "len_fb_correct": None if fb_correct is None else int(len(fb_correct)),
        "len_markov_correct": int(len(markov_correct)),
        "expected_wb": None if expected is None else expected["wb"],
        "expected_fb": None if expected is None else expected["fb"],
        "comparison": args.comparison,
    }
    if int(wb_counts.sum()) != len(wb_nn):
        raise ValueError(f"WB count total does not match NN vector: {validations}")
    if int(wb_counts.sum()) != len(wb_exact):
        raise ValueError(f"WB count total does not match exact vector: {validations}")
    if fb_correct is not None and int(fb_counts.sum()) != len(fb_correct):
        raise ValueError(f"FB count total does not match FB vector: {validations}")
    if int(fb_counts.sum()) != len(markov_correct):
        raise ValueError(f"FB count total does not match Markov vector: {validations}")
    if expected is not None:
        if len(wb_nn) != expected["wb"] or len(markov_correct) != expected["fb"]:
            raise ValueError(f"Observed lengths do not match expected handoff totals: {validations}")

    too_large = wb_counts > fb_counts
    validations["n_sequences_wb_count_gt_fb_count"] = int(too_large.sum())
    validations["max_wb_minus_fb_count"] = int((wb_counts - fb_counts).max())
    validations["sum_wb_actions_not_in_common_tail"] = int(np.maximum(wb_counts - fb_counts, 0).sum())

    if args.match_policy == "strict_wb" and np.any(too_large):
        first = int(np.where(too_large)[0][0])
        validation_path = out_dir / f"{paths.period}_analysis2_validations.json"
        validation_path.write_text(json.dumps(validations, indent=2), encoding="utf-8")
        raise ValueError(
            "Strict WB matching is impossible because some WB per-sequence counts exceed "
            "FB evaluated counts. First failure sequence "
            f"{first}: K={int(wb_counts[first])}, FB={int(fb_counts[first])}. "
            f"Validation details written to {validation_path}."
        )

    if args.match_policy == "common_tail":
        seq_counts = np.minimum(wb_counts, fb_counts)
    else:
        seq_counts = wb_counts

    wb_nn_sums = restrict_last_k_sums(wb_nn, wb_counts, seq_counts)
    wb_exact_sums = restrict_last_k_sums(wb_exact, wb_counts, seq_counts)
    markov_matched_sums = restrict_last_k_sums(markov_correct, fb_counts, seq_counts)
    fb_matched_sums = None
    if fb_correct is not None:
        fb_matched_sums = restrict_last_k_sums(fb_correct, fb_counts, seq_counts)

    seq_path = out_dir / f"{paths.period}_analysis2_sequence_summaries.npz"
    summary_payload = {
        "seq_counts": seq_counts,
        "wb_nn_sums": wb_nn_sums,
        "wb_exact_sums": wb_exact_sums,
        "markov_matched_sums": markov_matched_sums,
        "fb_original_counts": fb_counts,
        "wb_original_counts": wb_counts,
        "match_policy": np.array(args.match_policy),
        "comparison": np.array(args.comparison),
    }
    if fb_matched_sums is not None:
        summary_payload["fb_matched_sums"] = fb_matched_sums
    np.savez_compressed(seq_path, **summary_payload)

    rows = [
        {
            "period": paths.period,
            "model": "WB",
            "metric": "semantic_nn_mapped_accuracy",
            "match_policy": args.match_policy,
            "n_sequences": len(seq_counts),
            "n_actions": int(seq_counts.sum()),
            "accuracy": pooled_mean(wb_nn_sums, seq_counts),
        },
        {
            "period": paths.period,
            "model": "WB",
            "metric": "exact_match_accuracy_pre_mapping",
            "match_policy": args.match_policy,
            "n_sequences": len(seq_counts),
            "n_actions": int(seq_counts.sum()),
            "accuracy": pooled_mean(wb_exact_sums, seq_counts),
        },
        {
            "period": paths.period,
            "model": "Markov",
            "metric": "matched_tail_accuracy",
            "match_policy": args.match_policy,
            "n_sequences": len(seq_counts),
            "n_actions": int(seq_counts.sum()),
            "accuracy": pooled_mean(markov_matched_sums, seq_counts),
        },
    ]
    if fb_matched_sums is not None:
        rows.insert(
            2,
            {
                "period": paths.period,
                "model": "FB",
                "metric": "matched_tail_accuracy",
                "match_policy": args.match_policy,
                "n_sequences": len(seq_counts),
                "n_actions": int(seq_counts.sum()),
                "accuracy": pooled_mean(fb_matched_sums, seq_counts),
            },
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{paths.period}_analysis2_matched_accuracy.csv", index=False)

    validation_path = out_dir / f"{paths.period}_analysis2_validations.json"
    validation_path.write_text(json.dumps(validations, indent=2), encoding="utf-8")
    return df


def analysis3(paths: AnalysisPaths, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    seq_path = out_dir / f"{paths.period}_analysis2_sequence_summaries.npz"
    if not seq_path.exists():
        analysis2(paths, args, out_dir)
    data = np.load(seq_path)
    seq_counts = data["seq_counts"]

    specs = [
        ("WB", "semantic_nn_mapped_accuracy", data["wb_nn_sums"]),
        ("WB", "exact_match_accuracy_pre_mapping", data["wb_exact_sums"]),
        ("Markov", "matched_tail_accuracy", data["markov_matched_sums"]),
    ]
    if args.comparison == "full":
        if "fb_matched_sums" not in data:
            raise ValueError(f"Full comparison requested but {seq_path} has no fb_matched_sums")
        specs.insert(2, ("FB", "matched_tail_accuracy", data["fb_matched_sums"]))
    rows = []
    for model, metric, seq_sums in specs:
        boot_mean, ci_low, ci_high = cluster_bootstrap_ci(
            seq_sums=seq_sums,
            seq_counts=seq_counts,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        rows.append(
            {
                "period": paths.period,
                "model": model,
                "metric": metric,
                "match_policy": args.match_policy,
                "n_sequences": len(seq_counts),
                "n_actions": int(seq_counts.sum()),
                "raw_accuracy": pooled_mean(seq_sums, seq_counts),
                "bootstrap_mean": boot_mean,
                "ci_2.5": ci_low,
                "ci_97.5": ci_high,
                "n_bootstrap": args.n_bootstrap,
                "cluster_unit": "tokenized_dataset_test sequence",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{paths.period}_analysis3_cluster_bootstrap_ci.csv", index=False)
    return df


def write_combined_json(out_dir: Path, period: str, tables: dict[str, pd.DataFrame]) -> None:
    payload = {name: df.to_dict(orient="records") for name, df in tables.items()}
    (out_dir / f"{period}_all_analyses.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=["analysis1", "analysis2", "analysis3", "all"])
    parser.add_argument("--period", choices=sorted(PERIOD_LAYOUT), default="in_sample")
    parser.add_argument("--cache-root", type=Path, default=Path("/hdd/shared/AHRQ_LLM/cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/posthoc"))
    parser.add_argument("--wb-model-dir", default="llama3-WPE_EHRlogs-WB-FBeval-T3")
    parser.add_argument("--fb-model-dir", default="llama3-WPE_EHRlogs-FB-T3")
    parser.add_argument("--wb-tokenized-subdir", default=None)
    parser.add_argument("--fb-tokenized-subdir", default=None)
    parser.add_argument("--wb-metrics-subdir", default=None)
    parser.add_argument("--fb-metrics-subdir", default=None)
    parser.add_argument("--wb-prefix", default="llm")
    parser.add_argument("--fb-prefix", default="llm")
    parser.add_argument("--comparison", choices=["full", "wb_markov"], default="full")
    parser.add_argument("--skip-n-tokens", type=int, default=10)
    parser.add_argument("--force-counts", action="store_true")
    parser.add_argument("--skip-expected-lengths", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match-policy",
        choices=["strict_wb", "common_tail"],
        default="strict_wb",
        help=(
            "strict_wb keeps the full WB denominator and fails if any K_i exceeds FB_i. "
            "common_tail uses min(K_i, FB_i) and truncates WB, FB, and Markov to the shared suffix."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = build_paths(args)

    tables: dict[str, pd.DataFrame] = {}
    if args.analysis in {"analysis1", "all"}:
        tables["analysis1"] = analysis1(paths, args, out_dir)
        print(tables["analysis1"].to_string(index=False))
    if args.analysis in {"analysis2", "all"}:
        tables["analysis2"] = analysis2(paths, args, out_dir)
        print(tables["analysis2"].to_string(index=False))
    if args.analysis in {"analysis3", "all"}:
        tables["analysis3"] = analysis3(paths, args, out_dir)
        print(tables["analysis3"].to_string(index=False))
    if args.analysis == "all":
        write_combined_json(out_dir, args.period, tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
