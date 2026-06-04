"""Run the MSWD-only bootstrap experiment for Model A.

For each beta, this script generates one Model A data set with the same default
settings as ../main.py, computes the observed MSWD statistic, and then runs
B bootstrap replications for the MSWD null distribution.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyCode.dataGenerate import dataGenerate
from pyCode.global_var import device
from pyCode.maxSlicedWD_L0_L1Approx import (
    maxSlicedWDL0_L1Approx,
    maxSlicedWDL0_L1Approx_bootstrap,
    tune_l0,
)


DEFAULT_BETAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RAW_COLUMNS = [
    "model",
    "beta",
    "signal",
    "bootstrap_id",
    "observed_Tn",
    "bootstrap_mswd",
    "threshold",
    "reject",
    "true_label",
    "selected_lam_l0",
    "selected_lam_l1",
    "n1",
    "n2",
    "sample_dim",
    "alpha",
    "B",
    "seed",
    "runtime_seconds",
    "started_at",
    "finished_at",
]
SUMMARY_COLUMNS = [
    "model",
    "beta",
    "signal",
    "observed_Tn",
    "threshold",
    "pvalue",
    "reject",
    "true_label",
    "selected_lam_l0",
    "selected_lam_l1",
    "bootstrap_mean",
    "bootstrap_sd",
    "bootstrap_min",
    "bootstrap_max",
    "n1",
    "n2",
    "sample_dim",
    "alpha",
    "B",
    "seed",
    "runtime_seconds",
    "started_at",
    "finished_at",
]


def beta_key(beta: float) -> str:
    return f"{beta:.10g}"


def as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to("cpu").item())
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSWD-only Model A bootstrap experiment.")
    parser.add_argument("--beta", nargs="+", type=float, default=DEFAULT_BETAS, help="Beta values to run.")
    parser.add_argument("--bootstrap", "-B", type=int, default=500, help="Number of bootstrap replications per beta.")
    parser.add_argument("--n1", type=int, default=250)
    parser.add_argument("--n2", type=int, default=250)
    parser.add_argument("--sample-dim", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1919810)
    parser.add_argument("--force", action="store_true", help="Re-run beta values that already have saved results.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "mswd_modelA_bootstrap.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "mswd_modelA_summary.csv",
    )
    return parser.parse_args()


def read_completed_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            key = beta_key(float(row["beta"]))
            counts[key] = counts.get(key, 0) + 1
    return counts


def remove_beta_rows(path: Path, beta: float, columns: list[str]) -> None:
    if not path.exists():
        return
    key = beta_key(beta)
    kept_rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if beta_key(float(row["beta"])) != key:
                kept_rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(kept_rows)


def append_rows(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def run_one_beta(args: argparse.Namespace, beta: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = args.seed + int(round(beta * 1000)) * 1000
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    model = "mean-decay"
    signal_value = 0.8 * beta
    signal = torch.tensor([signal_value]).float()
    lam_l0_seq = torch.tensor([1, 5, 10, 20, 50])

    print(f"beta={beta}: generate Model A data, signal={signal_value}")
    data1, data2 = dataGenerate(args.n1, args.n2, args.sample_dim, signal, model, s=None)
    data1 = data1.to(device)
    data2 = data2.to(device)
    p1 = torch.ones(args.n1, device=device) / args.n1
    p2 = torch.ones(args.n2, device=device) / args.n2
    scale = (args.n1 * args.n2 / (args.n1 + args.n2)) ** 0.5

    print(f"beta={beta}: tune L0 sparsity parameter")
    lam_l0 = tune_l0(data1, data2, lam_l0_seq, k=2, reps=5)
    lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_l0**0.5), steps=10))

    print(f"beta={beta}: compute observed MSWD statistic")
    mswd_l0, _, lam_l1_selected = maxSlicedWDL0_L1Approx(
        data1,
        data2,
        p1,
        p2,
        lam_l0,
        lam_l1,
        candidate_adaptive=True,
        n_l1=10,
        reps=10,
    )
    observed_Tn = scale * mswd_l0

    print(f"beta={beta}: run MSWD bootstrap, B={args.bootstrap}")
    threshold, bootstrap_samples = maxSlicedWDL0_L1Approx_bootstrap(
        data1,
        data2,
        lam_l0,
        lam_l1,
        candidate_adaptive=True,
        n_l1=10,
        reps=1,
        B=args.bootstrap,
        alpha=args.alpha,
    )

    observed_value = as_float(observed_Tn)
    threshold_value = as_float(threshold)
    bootstrap_values = bootstrap_samples.detach().to("cpu").numpy().astype(float)
    pvalue = float(np.mean(bootstrap_values >= observed_value))
    reject = int(observed_value > threshold_value)
    true_label = int(beta > 0)
    runtime_seconds = time.time() - start_time
    finished_at = datetime.now().isoformat(timespec="seconds")

    common = {
        "model": "ModelA_mean_decay",
        "beta": beta,
        "signal": signal_value,
        "observed_Tn": observed_value,
        "threshold": threshold_value,
        "reject": reject,
        "true_label": true_label,
        "selected_lam_l0": as_float(lam_l0),
        "selected_lam_l1": as_float(lam_l1_selected),
        "n1": args.n1,
        "n2": args.n2,
        "sample_dim": args.sample_dim,
        "alpha": args.alpha,
        "B": args.bootstrap,
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    raw_rows = []
    for i, value in enumerate(bootstrap_values, start=1):
        row = dict(common)
        row["bootstrap_id"] = i
        row["bootstrap_mswd"] = float(value)
        raw_rows.append(row)

    summary_row = dict(common)
    summary_row.update(
        {
            "pvalue": pvalue,
            "bootstrap_mean": float(np.mean(bootstrap_values)),
            "bootstrap_sd": float(np.std(bootstrap_values, ddof=1)) if len(bootstrap_values) > 1 else 0.0,
            "bootstrap_min": float(np.min(bootstrap_values)),
            "bootstrap_max": float(np.max(bootstrap_values)),
        }
    )
    return raw_rows, summary_row


def main() -> None:
    args = parse_args()
    completed = read_completed_counts(args.out)

    print("MSWD-only Model A bootstrap experiment")
    print(f"device: {device}")
    print(f"beta values: {args.beta}")
    print(f"bootstrap replications per beta: {args.bootstrap}")
    print(f"raw output: {args.out}")
    print(f"summary output: {args.summary}")

    for beta in args.beta:
        key = beta_key(beta)
        if args.force:
            remove_beta_rows(args.out, beta, RAW_COLUMNS)
            remove_beta_rows(args.summary, beta, SUMMARY_COLUMNS)
        elif completed.get(key, 0) >= args.bootstrap:
            print(f"beta={beta}: already has {completed[key]} bootstrap rows; skipped.")
            continue

        raw_rows, summary_row = run_one_beta(args, beta)
        append_rows(args.out, raw_rows, RAW_COLUMNS)
        append_rows(args.summary, [summary_row], SUMMARY_COLUMNS)
        print(
            f"beta={beta}: saved. observed_Tn={summary_row['observed_Tn']:.6g}, "
            f"threshold={summary_row['threshold']:.6g}, pvalue={summary_row['pvalue']:.6g}, "
            f"reject={summary_row['reject']}"
        )


if __name__ == "__main__":
    main()
