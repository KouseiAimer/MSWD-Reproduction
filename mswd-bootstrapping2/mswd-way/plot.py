"""Plot MSWD-only Model A bootstrap results.

The black line is the observed statistic Tn from the original data, repeated
over the bootstrap index. The blue line is the 500 bootstrap MSWD statistics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "results" / "mswd_modelA_bootstrap.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures" / "mswd_modelA_bootstrap.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot MSWD bootstrap results for Model A.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_results(path: Path) -> dict[float, list[dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(f"Result file not found: {path}")

    grouped: dict[float, list[dict[str, float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            beta = float(row["beta"])
            grouped.setdefault(beta, []).append(
                {
                    "bootstrap_id": float(row["bootstrap_id"]),
                    "observed_Tn": float(row["observed_Tn"]),
                    "bootstrap_mswd": float(row["bootstrap_mswd"]),
                    "threshold": float(row["threshold"]),
                    "reject": float(row["reject"]),
                    "true_label": float(row["true_label"]),
                }
            )
    for beta in grouped:
        grouped[beta].sort(key=lambda item: item["bootstrap_id"])
    return grouped


def main() -> None:
    args = parse_args()
    grouped = load_results(args.input)
    betas = sorted(grouped)
    if not betas:
        raise ValueError(f"No rows found in {args.input}")

    n_panels = len(betas)
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for ax, beta in zip(axes_flat, betas):
        rows = grouped[beta]
        x = np.array([row["bootstrap_id"] for row in rows], dtype=float)
        observed = np.array([row["observed_Tn"] for row in rows], dtype=float)
        bootstrap = np.array([row["bootstrap_mswd"] for row in rows], dtype=float)
        threshold = rows[0]["threshold"]
        reject = int(rows[0]["reject"])
        pvalue = float(np.mean(bootstrap >= observed[0]))

        ax.plot(x, observed, color="black", linewidth=1.8, label="Observed Tn")
        ax.plot(x, bootstrap, color="#1f77b4", linewidth=1.1, alpha=0.9, label="Bootstrap MSWD")
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1.0, label="95% threshold")
        ax.set_title(f"beta = {beta:g}, reject = {reject}, p = {pvalue:.3f}")
        ax.set_xlabel("Bootstrap index")
        ax.set_ylabel("Statistic")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("Model A: Observed MSWD Statistic vs Bootstrap Statistics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Figure saved to: {args.output}")
    print(f"PDF saved to: {pdf_path}")


if __name__ == "__main__":
    main()
