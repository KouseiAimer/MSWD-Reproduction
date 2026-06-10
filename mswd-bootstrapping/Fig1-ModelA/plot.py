"""Plot Model A first-50 rejection rates produced by stat.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE / "modelA_first50_summary.csv"
DEFAULT_OUTPUT = BASE / "Figure1_ModelA_first50.png"

METHODS = [
    ("Proposed", "proposed_rate", "black", "o", "-"),
    ("MMD-G", "mmd_g_rate", "#d62728", "s", "--"),
    ("ED2", "ed2_rate", "#2ca02c", "^", "-."),
    ("BG", "bg_rate", "#9467bd", "D", ":"),
    ("PW", "pw_rate", "#1f77b4", "v", "-"),
]


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    rows.sort(key=lambda row: float(row["beta"]))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Model A first-50 summary.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_summary(args.input)
    betas = [float(row["beta"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, column, color, marker, linestyle in METHODS:
        values = [float(row[column]) for row in rows]
        ax.plot(
            betas,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            markersize=6,
            label=label,
        )

    ax.set_title("Model A", fontsize=13)
    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_ylabel("Empirical rejection rate", fontsize=12)
    ax.set_xticks(betas)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.75)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {args.output}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
