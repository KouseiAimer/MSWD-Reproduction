"""Checkpointed Figure 1 Model A simulation.

This file is created from the logic of ../main.py, but does not modify it.

Default target:
    Model A, p = 500, n1 = n2 = 250
    beta in {0, 0.2, 0.4, 0.6, 0.8, 1}
    50 completed Monte Carlo runs per beta

Each completed run is appended to results/model_A_raw_results.csv immediately.
If a run is interrupted before it is saved, the next execution resumes from
that same run_id.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyCode.dataGenerate import dataGenerate  # noqa: E402
from pyCode.global_var import device  # noqa: E402
from pyCode.maxSlicedWD_L0_L1Approx import (  # noqa: E402
    maxSlicedWDL0_L1Approx,
    maxSlicedWDL0_L1Approx_bootstrap,
    tune_l0,
)
from pyCode.MMD import kernelSigma  # noqa: E402
from pyCode.pwdPerm import pwdPerm  # noqa: E402


BETA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CSV_COLUMNS = [
    "model",
    "beta",
    "signal",
    "run_id",
    "seed",
    "n1",
    "n2",
    "sample_dim",
    "alpha",
    "proposed_B",
    "perm_nperm",
    "pw_nperm",
    "tune_reps",
    "reps",
    "selected_lam_l0",
    "proposed_statistic",
    "proposed_threshold",
    "proposed_pvalue",
    "proposed_reject",
    "mmd_g_pvalue",
    "mmd_g_reject",
    "mmd_l_pvalue",
    "mmd_l_reject",
    "ed2_pvalue",
    "ed2_reject",
    "bg_pvalue",
    "bg_reject",
    "pw_pvalue",
    "pw_reject",
    "runtime_seconds",
    "started_at",
    "finished_at",
]


def beta_key(beta: float) -> str:
    return f"{beta:.10g}"


def parse_betas(raw: list[str] | None) -> list[float]:
    if not raw:
        return BETA_GRID
    if len(raw) == 1 and raw[0].lower() == "all":
        return BETA_GRID
    return [float(value) for value in raw]


def tensor_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to("cpu").item())
    return float(value)


def tensor_bool_int(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(bool(value.detach().to("cpu").item()))
    return int(bool(value))


def read_completed_runs(path: Path) -> dict[str, set[int]]:
    completed: dict[str, set[int]] = {}
    if not path.exists():
        return completed

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            required = [
                "proposed_reject",
                "mmd_g_reject",
                "mmd_l_reject",
                "ed2_reject",
                "bg_reject",
                "pw_reject",
            ]
            if all(row.get(name, "") != "" for name in required):
                completed.setdefault(beta_key(float(row["beta"])), set()).add(int(row["run_id"]))
    return completed


def append_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})
        f.flush()


def comparison_perm_test(data1: torch.Tensor, data2: torch.Tensor, nperm: int = 500, alpha: float = 0.05) -> dict[str, torch.Tensor]:
    """Permutation tests used in Figure 1: MMD-G, MMD-L, ED2 and BG."""
    data1 = data1.to(device)
    data2 = data2.to(device)
    n1 = data1.size(0)
    n2 = data2.size(0)
    n = n1 + n2

    dist1 = torch.cdist(data1, data1, p=2)
    dist2 = torch.cdist(data2, data2, p=2)
    dist12 = torch.cdist(data1, data2, p=2)
    ed2 = 2 * torch.mean(dist12) - torch.sum(dist1) / n1 / (n1 - 1) - torch.sum(dist2) / n2 / (n2 - 1)
    dist_mat2 = torch.cat((torch.cat((dist1, dist12), 1), torch.cat((dist12.T, dist2), 1)), 0)

    t_scale2 = torch.sum(dist2) / n2 / (n2 - 1) - torch.sum(dist1) / n1 / (n1 - 1)
    bg = (ed2**2 + t_scale2**2) / 2

    sigma = torch.from_numpy(kernelSigma(data1, data2)).to(device)
    ker_mat_1 = torch.exp(-torch.cdist(data1, data1) ** 2 / sigma**2)
    ker_mat_2 = torch.exp(-torch.cdist(data2, data2) ** 2 / sigma**2)
    ker_mat_12 = torch.exp(-torch.cdist(data1, data2) ** 2 / sigma**2)
    ker_mat_1 = ker_mat_1 - torch.diag(torch.diag(ker_mat_1))
    ker_mat_2 = ker_mat_2 - torch.diag(torch.diag(ker_mat_2))
    mmd = torch.sum(ker_mat_1) / n1 / (n1 - 1) + torch.sum(ker_mat_2) / n2 / (n2 - 1) - 2 * torch.mean(ker_mat_12)
    ker_mat = torch.cat((torch.cat((ker_mat_1, ker_mat_12), 1), torch.cat((ker_mat_12.T, ker_mat_2), 1)), 0)

    lapker_mat_1 = torch.exp(-dist1 / sigma)
    lapker_mat_2 = torch.exp(-dist2 / sigma)
    lapker_mat_12 = torch.exp(-dist12 / sigma)
    lapker_mat_1 = lapker_mat_1 - torch.diag(torch.diag(lapker_mat_1))
    lapker_mat_2 = lapker_mat_2 - torch.diag(torch.diag(lapker_mat_2))
    lapmmd = torch.sum(lapker_mat_1) / n1 / (n1 - 1) + torch.sum(lapker_mat_2) / n2 / (n2 - 1) - 2 * torch.mean(lapker_mat_12)
    lapker_mat = torch.cat((torch.cat((lapker_mat_1, lapker_mat_12), 1), torch.cat((lapker_mat_12.T, lapker_mat_2), 1)), 0)

    ed2_perm = torch.zeros(nperm, device=device)
    bg_perm = torch.zeros(nperm, device=device)
    mmd_perm = torch.zeros(nperm, device=device)
    lapmmd_perm = torch.zeros(nperm, device=device)
    for perm_id in range(nperm):
        if (perm_id + 1) % 100 == 0:
            print("run {} of permutation".format(perm_id + 1))
        locperm = torch.randperm(n, device=device)
        loc = locperm[0:n1]
        loc2 = locperm[n1:n]

        dist_mat11 = dist_mat2[loc, :][:, loc]
        dist_mat12 = dist_mat2[loc, :][:, loc2]
        dist_mat22 = dist_mat2[loc2, :][:, loc2]
        ed2_perm[perm_id] = 2 * torch.mean(dist_mat12) - torch.sum(dist_mat11) / n1 / (n1 - 1) - torch.sum(dist_mat22) / n2 / (n2 - 1)

        t_scale2_perm = torch.sum(dist_mat22) / n2 / (n2 - 1) - torch.sum(dist_mat11) / n1 / (n1 - 1)
        bg_perm[perm_id] = (ed2_perm[perm_id] ** 2 + t_scale2_perm**2) / 2

        ker_mat11 = ker_mat[loc, :][:, loc]
        ker_mat22 = ker_mat[loc2, :][:, loc2]
        ker_mat12 = ker_mat[loc, :][:, loc2]
        mmd_perm[perm_id] = torch.sum(ker_mat11) / n1 / (n1 - 1) + torch.sum(ker_mat22) / n2 / (n2 - 1) - 2 * torch.mean(ker_mat12)

        lapker_mat11 = lapker_mat[loc, :][:, loc]
        lapker_mat22 = lapker_mat[loc2, :][:, loc2]
        lapker_mat12 = lapker_mat[loc, :][:, loc2]
        lapmmd_perm[perm_id] = torch.sum(lapker_mat11) / n1 / (n1 - 1) + torch.sum(lapker_mat22) / n2 / (n2 - 1) - 2 * torch.mean(lapker_mat12)

    return {
        "mmd_pval": torch.mean((mmd_perm > mmd).float()),
        "mmd_decision": mmd > torch.quantile(mmd_perm, 1 - alpha),
        "lapmmd_pval": torch.mean((lapmmd_perm > lapmmd).float()),
        "lapmmd_decision": lapmmd > torch.quantile(lapmmd_perm, 1 - alpha),
        "edl2_pval": torch.mean((ed2_perm > ed2).float()),
        "edl2_decision": ed2 > torch.quantile(ed2_perm, 1 - alpha),
        "bg_pval": torch.mean((bg_perm > bg).float()),
        "bg_decision": bg > torch.quantile(bg_perm, 1 - alpha),
    }


def run_one(args: argparse.Namespace, beta: float, run_id: int) -> dict[str, object]:
    seed = args.seed_base + int(round(beta * 1000)) * 100000 + run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    # Paper Model A has mu_j = 0.8 * beta * j^(-3). The local generator
    # uses mean = signal / j^3, so signal must be 0.8 * beta.
    signal = 0.8 * beta
    data1, data2 = dataGenerate(
        args.n1,
        args.n2,
        args.sample_dim,
        torch.tensor([signal]).float(),
        "mean-decay",
        s=None,
    )
    data1 = data1.to(device)
    data2 = data2.to(device)
    p1 = torch.ones(args.n1, device=device) / args.n1
    p2 = torch.ones(args.n2, device=device) / args.n2
    scale = math.sqrt(args.n1 * args.n2 / (args.n1 + args.n2))

    lam_l0_seq = torch.tensor([1, 5, 10, 20, 50])
    lam_l0 = tune_l0(data1, data2, lam_l0_seq, k=2, reps=args.tune_reps)
    lam_l0_float = tensor_float(lam_l0)
    lam_l1 = torch.exp(torch.linspace(np.log(1), np.log(lam_l0_float**0.5), steps=10))
    mswd_l0, _, _ = maxSlicedWDL0_L1Approx(
        data1,
        data2,
        p1,
        p2,
        lam_l0,
        lam_l1,
        candidate_adaptive=True,
        n_l1=10,
        reps=args.reps,
    )
    proposed_statistic = scale * mswd_l0
    proposed_threshold, proposed_boots = maxSlicedWDL0_L1Approx_bootstrap(
        data1,
        data2,
        lam_l0,
        lam_l1,
        candidate_adaptive=True,
        n_l1=10,
        reps=1,
        B=args.proposed_B,
        alpha=args.alpha,
    )
    proposed_pvalue = torch.mean((proposed_boots > proposed_statistic).float())
    proposed_reject = proposed_statistic > proposed_threshold

    perm = comparison_perm_test(data1, data2, nperm=args.perm_nperm, alpha=args.alpha)
    pw = pwdPerm(data1, data2, alpha=args.alpha, n_perm=args.pw_nperm)

    runtime_seconds = time.time() - start_time
    finished_at = datetime.now().isoformat(timespec="seconds")

    return {
        "model": "A_mean_decay",
        "beta": beta,
        "signal": signal,
        "run_id": run_id,
        "seed": seed,
        "n1": args.n1,
        "n2": args.n2,
        "sample_dim": args.sample_dim,
        "alpha": args.alpha,
        "proposed_B": args.proposed_B,
        "perm_nperm": args.perm_nperm,
        "pw_nperm": args.pw_nperm,
        "tune_reps": args.tune_reps,
        "reps": args.reps,
        "selected_lam_l0": lam_l0_float,
        "proposed_statistic": tensor_float(proposed_statistic),
        "proposed_threshold": tensor_float(proposed_threshold),
        "proposed_pvalue": tensor_float(proposed_pvalue),
        "proposed_reject": tensor_bool_int(proposed_reject),
        "mmd_g_pvalue": tensor_float(perm["mmd_pval"]),
        "mmd_g_reject": tensor_bool_int(perm["mmd_decision"]),
        "mmd_l_pvalue": tensor_float(perm["lapmmd_pval"]),
        "mmd_l_reject": tensor_bool_int(perm["lapmmd_decision"]),
        "ed2_pvalue": tensor_float(perm["edl2_pval"]),
        "ed2_reject": tensor_bool_int(perm["edl2_decision"]),
        "bg_pvalue": tensor_float(perm["bg_pval"]),
        "bg_reject": tensor_bool_int(perm["bg_decision"]),
        "pw_pvalue": tensor_float(pw["kPWD_perm_pval"]),
        "pw_reject": tensor_bool_int(pw["kPWD_perm_decision"]),
        "runtime_seconds": runtime_seconds,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run checkpointed Figure 1 Model A simulations.")
    parser.add_argument("--beta", nargs="+", default=None, help="Beta value(s), e.g. --beta 0.2 or --beta all.")
    parser.add_argument("--nrun", type=int, default=50, help="Target completed runs per beta.")
    parser.add_argument("--n1", type=int, default=250)
    parser.add_argument("--n2", type=int, default=250)
    parser.add_argument("--sample_dim", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--proposed-B", dest="proposed_B", type=int, default=300)
    parser.add_argument("--perm-nperm", type=int, default=500)
    parser.add_argument("--pw-nperm", type=int, default=500)
    parser.add_argument("--tune-reps", type=int, default=5)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=20260602)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "model_A_raw_results.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    betas = parse_betas(args.beta)
    completed = read_completed_runs(args.out)

    print("Figure 1 Model A simulation")
    print(f"device: {device}")
    print(f"raw result file: {args.out}")
    print(f"beta values: {betas}")
    print(f"target completed runs per beta: {args.nrun}")

    new_runs = 0
    for beta in betas:
        key = beta_key(beta)
        done = completed.get(key, set())
        if len(done) >= args.nrun:
            print(f"beta={beta}: already completed {len(done)}/{args.nrun}; no more runs needed.")
            continue

        print(f"beta={beta}: completed {len(done)}/{args.nrun}; continuing.")
        for run_id in range(1, args.nrun + 1):
            if run_id in done:
                continue

            print(f"beta={beta}, run={run_id}/{args.nrun}: start")
            row = run_one(args, beta, run_id)
            append_result(args.out, row)
            done.add(run_id)
            completed.setdefault(key, set()).add(run_id)
            new_runs += 1
            print(
                f"beta={beta}, run={run_id}/{args.nrun}: saved; "
                f"Proposed={row['proposed_reject']}, MMD-G={row['mmd_g_reject']}, "
                f"MMD-L={row['mmd_l_reject']}, ED2={row['ed2_reject']}, "
                f"BG={row['bg_reject']}, PW={row['pw_reject']}; "
                f"runtime={row['runtime_seconds']:.1f}s"
            )

        print(f"beta={beta}: completed {len(done)}/{args.nrun}.")

    if new_runs == 0:
        print("No new simulation runs were needed.")


if __name__ == "__main__":
    main()
