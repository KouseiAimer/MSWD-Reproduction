"""Compute Model A rejection rates from main.py log files.

The console logs contain cumulative rejection rates within each separate
process. Files such as results_beta0_1.txt and results_beta0_2.txt both start
from run 0, so their cumulative rates must be combined by counts, not by simply
continuing the displayed cumulative rate.

This script keeps only the first 50 complete runs for each beta. A complete run
is defined as a run that reaches the final default output line in main.py:
    run k, percentage of rejection: pwd perm k3 ...
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = Path(__file__).resolve().parent
DEFAULT_NRUN = 50

METHODS = [
    ("Proposed", "proposed"),
    ("MMD-G", "mmd_g"),
    ("ED2", "ed2"),
    ("BG", "bg"),
    ("PW", "pw"),
    ("MSWD-L1_logged", "mswd_l1"),
    ("sMMD_logged", "smmd"),
]

PLOT_METHOD_KEYS = ["proposed", "mmd_g", "ed2", "bg", "pw"]


def read_text(path: Path) -> str:
    data = path.read_bytes()
    if not data:
        return ""
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="ignore")
    for encoding in ("utf-8", "utf-16", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def beta_from_filename(path: Path) -> tuple[float, int]:
    match = re.match(r"^results_beta(.+?)(?:_(\d+))?\.txt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse beta from filename: {path.name}")
    beta = float(match.group(1))
    run_part = int(match.group(2) or 0)
    return beta, run_part


def parse_rates(path: Path) -> dict[str, Any]:
    text = read_text(path)
    beta, run_part = beta_from_filename(path)
    complete_runs = sorted(
        set(
            int(run)
            for run in re.findall(
                r"(?m)^run\s+(\d+),\s*percentage of rejection:\s*pwd perm k3", text
            )
        )
    )

    rates: dict[str, dict[int, float]] = {
        "proposed": {
            int(run): float(rate)
            for run, rate in re.findall(
                r"(?m)^run\s+(\d+),\s*percentage of rejection:\s*max-sliced wd with l0 sparsity\s+([0-9.eE+-]+)",
                text,
            )
        },
        "mswd_l1": {
            int(run): float(rate)
            for run, rate in re.findall(
                r"(?m)^run\s+(\d+),\s*percentage of rejection:\s*max-sliced wd with l1 sparsity\s+([0-9.eE+-]+)",
                text,
            )
        },
        "smmd": {
            int(run): float(rate)
            for run, rate in re.findall(
                r"(?m)^run\s+(\d+),\s*percentage of rejection:\s*smmd\s+([0-9.eE+-]+)",
                text,
            )
        },
        "pw": {
            int(run): float(rate)
            for run, rate in re.findall(
                r"(?m)^run\s+(\d+),\s*percentage of rejection:\s*pwd perm k3\s+([0-9.eE+-]+)",
                text,
            )
        },
    }

    perm_rates = {
        int(run): (float(mmd), float(ed2), float(bg))
        for run, mmd, ed2, bg in re.findall(
            r"run\s+(\d+),\s*percentage of rejection:\s*mmd\s+([0-9.eE+-]+)\s*,\s*edl1\s+[0-9.eE+-]+,\s*edl2\s+([0-9.eE+-]+),\s*bg\s+([0-9.eE+-]+)",
            text,
            flags=re.S,
        )
    }
    rates["mmd_g"] = {run: values[0] for run, values in perm_rates.items()}
    rates["ed2"] = {run: values[1] for run, values in perm_rates.items()}
    rates["bg"] = {run: values[2] for run, values in perm_rates.items()}

    complete_count = len(complete_runs)
    return {
        "file": path.name,
        "beta": beta,
        "run_part": run_part,
        "complete_count": complete_count,
        "complete_runs": complete_runs,
        "rates": rates,
        "size": path.stat().st_size,
    }


def count_rejections_from_prefix(rates_by_run: dict[int, float], n_used: int) -> int:
    if n_used == 0:
        return 0
    local_run = n_used - 1
    if local_run not in rates_by_run:
        raise RuntimeError(f"Missing cumulative rate for local run {local_run}")
    return int(round(rates_by_run[local_run] * n_used))


def decisions_from_prefix(rates_by_run: dict[int, float], n_used: int) -> list[int]:
    decisions: list[int] = []
    previous_count = 0
    for local_run in range(n_used):
        if local_run not in rates_by_run:
            raise RuntimeError(f"Missing cumulative rate for local run {local_run}")
        current_count = int(round(rates_by_run[local_run] * (local_run + 1)))
        decision = current_count - previous_count
        if decision not in (0, 1):
            raise RuntimeError(
                f"Invalid recovered decision={decision} at local run {local_run}; "
                f"rate={rates_by_run[local_run]}"
            )
        decisions.append(decision)
        previous_count = current_count
    return decisions


def summarize(parsed_files: list[dict[str, Any]], target_nrun: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for item in parsed_files:
        grouped[item["beta"]].append(item)
    for beta in grouped:
        grouped[beta].sort(key=lambda item: item["run_part"])

    contribution_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for beta in sorted(grouped):
        remaining = target_nrun
        totals = {key: 0 for _, key in METHODS}
        global_run = 1

        for item in grouped[beta]:
            if remaining <= 0:
                break
            take = min(remaining, item["complete_count"])
            if take <= 0:
                continue

            contribution = {
                "beta": beta,
                "source_file": item["file"],
                "complete_runs_in_file": item["complete_count"],
                "used_runs_for_first50": take,
            }
            method_decisions: dict[str, list[int]] = {}
            for _, key in METHODS:
                count = count_rejections_from_prefix(item["rates"][key], take)
                totals[key] += count
                contribution[f"{key}_reject_count"] = count
                contribution[f"{key}_rate_within_used"] = count / take
                method_decisions[key] = decisions_from_prefix(item["rates"][key], take)

            contribution_rows.append(contribution)

            for local_index in range(take):
                row = {
                    "beta": beta,
                    "global_run": global_run,
                    "source_file": item["file"],
                    "local_run": local_index,
                }
                for _, key in METHODS:
                    row[key] = method_decisions[key][local_index]
                decision_rows.append(row)
                global_run += 1

            remaining -= take

        n_used = target_nrun - remaining
        if n_used < target_nrun:
            raise RuntimeError(f"beta={beta:g} has only {n_used} complete runs; need {target_nrun}.")

        summary = {
            "beta": beta,
            "n_used": n_used,
            "note": (
                "Rates are count-weighted across files. MMD-L cannot be recovered "
                "because main.py does not print lapmmd_decision."
            ),
        }
        for _, key in METHODS:
            summary[f"{key}_reject_count"] = totals[key]
            summary[f"{key}_rate"] = totals[key] / n_used
        summary_rows.append(summary)

    return contribution_rows, decision_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Statistically combine main.py result logs by beta.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--nrun", type=int, default=DEFAULT_NRUN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = sorted(args.root.glob("results_beta*.txt"), key=lambda path: beta_from_filename(path))
    parsed_files = [parse_rates(path) for path in files if path.name != "results_beta0.8txt"]
    parsed_files = [item for item in parsed_files if item["complete_count"] > 0]

    contribution_rows, decision_rows, summary_rows = summarize(parsed_files, args.nrun)

    contribution_columns = [
        "beta",
        "source_file",
        "complete_runs_in_file",
        "used_runs_for_first50",
    ]
    for _, key in METHODS:
        contribution_columns.extend([f"{key}_reject_count", f"{key}_rate_within_used"])

    decision_columns = ["beta", "global_run", "source_file", "local_run"] + [key for _, key in METHODS]

    summary_columns = ["beta", "n_used"]
    for _, key in METHODS:
        summary_columns.extend([f"{key}_reject_count", f"{key}_rate"])
    summary_columns.append("note")

    write_csv(args.outdir / "modelA_first50_file_contributions.csv", contribution_rows, contribution_columns)
    write_csv(args.outdir / "modelA_first50_decisions.csv", decision_rows, decision_columns)
    write_csv(args.outdir / "modelA_first50_summary.csv", summary_rows, summary_columns)

    print("Weighted first-50 summary:")
    for row in summary_rows:
        print(
            f"beta={float(row['beta']):g}, n={row['n_used']}, "
            f"Proposed={row['proposed_rate']:.3f}, MMD-G={row['mmd_g_rate']:.3f}, "
            f"ED2={row['ed2_rate']:.3f}, BG={row['bg_rate']:.3f}, PW={row['pw_rate']:.3f}, "
            f"MSWD-L1(logged)={row['mswd_l1_rate']:.3f}, sMMD(logged)={row['smmd_rate']:.3f}"
        )
    print(f"Saved: {args.outdir / 'modelA_first50_file_contributions.csv'}")
    print(f"Saved: {args.outdir / 'modelA_first50_decisions.csv'}")
    print(f"Saved: {args.outdir / 'modelA_first50_summary.csv'}")
    print("Note: MMD-L is not available in these txt logs because main.py never prints lapmmd_decision.")


if __name__ == "__main__":
    main()
