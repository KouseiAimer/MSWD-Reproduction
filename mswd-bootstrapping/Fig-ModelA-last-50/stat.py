"""Compute Model A rejection rates from the last 50 complete log runs.

Each results_beta*.txt file records cumulative rejection rates within that one
process, so files under the same beta must be converted to per-run 0/1
decisions before they can be combined. This script concatenates complete runs
by beta and file order, then keeps the last 50 complete runs for each beta.
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
    return float(match.group(1)), int(match.group(2) or 0)


def decisions_from_rates(rates_by_run: dict[int, float], complete_runs: list[int]) -> dict[int, int]:
    decisions: dict[int, int] = {}
    previous_count = 0
    for run_id in complete_runs:
        if run_id not in rates_by_run:
            raise RuntimeError(f"Missing cumulative rate for local run {run_id}")
        current_count = int(round(rates_by_run[run_id] * (run_id + 1)))
        decision = current_count - previous_count
        if decision not in (0, 1):
            raise RuntimeError(
                f"Invalid recovered decision={decision} at local run {run_id}; "
                f"rate={rates_by_run[run_id]}"
            )
        decisions[run_id] = decision
        previous_count = current_count
    return decisions


def parse_one_file(path: Path) -> list[dict[str, Any]]:
    beta, run_part = beta_from_filename(path)
    text = read_text(path)
    if not text:
        return []

    complete_runs = sorted(
        set(
            int(run)
            for run in re.findall(
                r"(?m)^run\s+(\d+),\s*percentage of rejection:\s*pwd perm k3",
                text,
            )
        )
    )
    if not complete_runs:
        return []

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

    decisions = {key: decisions_from_rates(rates[key], complete_runs) for _, key in METHODS}

    rows: list[dict[str, Any]] = []
    for local_order, run_id in enumerate(complete_runs, start=1):
        row = {
            "beta": beta,
            "run_part": run_part,
            "source_file": path.name,
            "local_order": local_order,
            "local_run": run_id,
        }
        for _, key in METHODS:
            row[key] = decisions[key][run_id]
        rows.append(row)
    return rows


def collect_last_runs(root: Path, nrun: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    files = sorted(root.glob("results_beta*.txt"), key=lambda path: beta_from_filename(path))
    for path in files:
        for row in parse_one_file(path):
            grouped[float(row["beta"])].append(row)

    decision_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for beta in sorted(grouped):
        all_rows = grouped[beta]
        if len(all_rows) < nrun:
            raise RuntimeError(f"beta={beta:g} has only {len(all_rows)} complete runs; need {nrun}.")
        selected = all_rows[-nrun:]

        file_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for global_run, row in enumerate(selected, start=1):
            out = dict(row)
            out["global_run"] = global_run
            decision_rows.append(out)
            file_groups[row["source_file"]].append(out)

        for source_file, rows in file_groups.items():
            contribution = {
                "beta": beta,
                "source_file": source_file,
                "used_runs_for_last50": len(rows),
                "first_used_local_run": min(int(row["local_run"]) for row in rows),
                "last_used_local_run": max(int(row["local_run"]) for row in rows),
            }
            for _, key in METHODS:
                reject_count = sum(int(row[key]) for row in rows)
                contribution[f"{key}_reject_count"] = reject_count
                contribution[f"{key}_rate_within_used"] = reject_count / len(rows)
            contribution_rows.append(contribution)

        summary = {
            "beta": beta,
            "n_used": nrun,
            "note": (
                "Last 50 complete runs after concatenating files by beta. "
                "MMD-L cannot be recovered because main.py does not print lapmmd_decision."
            ),
        }
        for _, key in METHODS:
            reject_count = sum(int(row[key]) for row in selected)
            summary[f"{key}_reject_count"] = reject_count
            summary[f"{key}_rate"] = reject_count / nrun
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
    parser = argparse.ArgumentParser(description="Summarize the last complete Model A runs by beta.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--nrun", type=int, default=DEFAULT_NRUN)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    contribution_rows, decision_rows, summary_rows = collect_last_runs(args.root, args.nrun)

    contribution_columns = [
        "beta",
        "source_file",
        "used_runs_for_last50",
        "first_used_local_run",
        "last_used_local_run",
    ]
    for _, key in METHODS:
        contribution_columns.extend([f"{key}_reject_count", f"{key}_rate_within_used"])

    decision_columns = [
        "beta",
        "global_run",
        "source_file",
        "local_order",
        "local_run",
    ] + [key for _, key in METHODS]

    summary_columns = ["beta", "n_used"]
    for _, key in METHODS:
        summary_columns.extend([f"{key}_reject_count", f"{key}_rate"])
    summary_columns.append("note")

    write_csv(args.outdir / "modelA_last50_file_contributions.csv", contribution_rows, contribution_columns)
    write_csv(args.outdir / "modelA_last50_decisions.csv", decision_rows, decision_columns)
    write_csv(args.outdir / "modelA_last50_summary.csv", summary_rows, summary_columns)

    print("Last-50 summary:")
    for row in summary_rows:
        print(
            f"beta={float(row['beta']):g}, n={row['n_used']}, "
            f"Proposed={row['proposed_rate']:.3f}, MMD-G={row['mmd_g_rate']:.3f}, "
            f"ED2={row['ed2_rate']:.3f}, BG={row['bg_rate']:.3f}, PW={row['pw_rate']:.3f}, "
            f"MSWD-L1(logged)={row['mswd_l1_rate']:.3f}, sMMD(logged)={row['smmd_rate']:.3f}"
        )
    print(f"Saved: {args.outdir / 'modelA_last50_file_contributions.csv'}")
    print(f"Saved: {args.outdir / 'modelA_last50_decisions.csv'}")
    print(f"Saved: {args.outdir / 'modelA_last50_summary.csv'}")


if __name__ == "__main__":
    main()
