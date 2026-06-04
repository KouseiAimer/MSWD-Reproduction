"""Plot Figure 1 Model A results from checkpointed simulation CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("proposed", "Proposed"),
    ("mmd_g", "MMD-G"),
    ("mmd_l", "MMD-L"),
    ("ed2", "ED2"),
    ("bg", "BG"),
    ("pw", "PW"),
]


def read_raw(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Raw result file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("model") != "A_mean_decay":
            continue
        grouped.setdefault(float(row["beta"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for beta in sorted(grouped):
        beta_rows = grouped[beta]
        item: dict[str, object] = {"beta": beta, "n_completed": len(beta_rows)}
        for method, _ in METHODS:
            values = [
                int(row[f"{method}_reject"])
                for row in beta_rows
                if row.get(f"{method}_reject", "") != ""
            ]
            item[method] = float(np.mean(values)) if values else float("nan")
        summary_rows.append(item)
    return summary_rows


def save_summary(path: Path, summary_rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["beta", "n_completed"] + [method for method, _ in METHODS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot(summary_rows: list[dict[str, object]], png_path: Path, pdf_path: Path) -> None:
    if not summary_rows:
        raise ValueError("No summary rows available for plotting.")

    styles = {
        "proposed": {"color": "red", "linestyle": "-", "marker": "o"},
        "mmd_g": {"color": "black", "linestyle": "--", "marker": "s"},
        "mmd_l": {"color": "dimgray", "linestyle": "--", "marker": "D"},
        "ed2": {"color": "tab:green", "linestyle": (0, (6, 3)), "marker": "v"},
        "bg": {"color": "tab:purple", "linestyle": "-.", "marker": "P"},
        "pw": {"color": "tab:orange", "linestyle": (0, (8, 3, 2, 3)), "marker": "X"},
    }

    betas = [float(row["beta"]) for row in summary_rows]
    plt.figure(figsize=(7.5, 5.2))
    for method, label in METHODS:
        y = [float(row[method]) for row in summary_rows]
        plt.plot(betas, y, label=label, linewidth=2, markersize=5, **styles[method])

    plt.xlabel(r"$\beta$")
    plt.ylabel("Empirical rejection rate")
    plt.title("Model A, p = 500")
    plt.ylim(-0.02, 1.02)
    plt.xlim(min(betas) - 0.03, max(betas) + 0.03)
    plt.grid(True, color="0.88", linewidth=0.8)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent / "results"
    parser = argparse.ArgumentParser(description="Plot Model A simulation results.")
    parser.add_argument("--raw", type=Path, default=base / "model_A_raw_results.csv")
    parser.add_argument("--summary", type=Path, default=base / "model_A_summary.csv")
    parser.add_argument("--png", type=Path, default=base / "figure1_model_A.png")
    parser.add_argument("--pdf", type=Path, default=base / "figure1_model_A.pdf")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_raw(args.raw)
    summary_rows = summarize(rows)
    save_summary(args.summary, summary_rows)
    plot(summary_rows, args.png, args.pdf)

    print(f"Saved summary: {args.summary}")
    print(f"Saved PNG: {args.png}")
    print(f"Saved PDF: {args.pdf}")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
