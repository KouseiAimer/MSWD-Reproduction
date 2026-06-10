"""Accelerated MSWD-only simulation runner.

This script intentionally skips all non-MSWD baselines from main.py.  The
default settings are faster than the paper settings: lambdas are tuned once and
then reused, bootstrap counts are smaller, and optimization restarts are fewer.
Use --tune_every 1 and larger B/reps values when you need stricter reproduction.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from pyCode import myProj
from pyCode.dataGenerate import dataGenerate
from pyCode.global_var import device, to_scalar
from pyCode.maxSlicedWD_L0_L1Approx import (
    maxSlicedWDL0_L1Approx,
    maxSlicedWDL0_L1Approx_bootstrap,
    tune_l0,
)
from pyCode.maxSlicedWD_L1 import maxSlicedWD, maxSlicedWD_bootstrap, tune_l1


def _log_grid(max_value, steps):
    max_value = torch.as_tensor(max_value, dtype=torch.float32, device=device)
    end = torch.log(torch.sqrt(max_value)).detach().cpu().item()
    return torch.exp(torch.linspace(0.0, end, steps=steps, device=device))


def _as_float(value):
    if value is None:
        return ""
    return float(to_scalar(value))


def _as_int(value):
    if value is None:
        return ""
    return int(float(to_scalar(value)))


def _should_tune(run_id, tune_every, current_value, fixed_value):
    if fixed_value is not None:
        return False
    if current_value is None:
        return True
    if tune_every <= 0:
        return False
    return run_id % tune_every == 0


def _prepare_signal(args):
    signal = torch.tensor(args.signal, dtype=torch.float32, device=device)
    if args.model == "joint":
        if args.sig == "sig5":
            return torch.ones(100, device=device)
        if args.sig == "sig4":
            return torch.ones(80, device=device)
        if args.sig == "sig3":
            return torch.ones(60, device=device)
        if args.sig == "sig2":
            return torch.ones(40, device=device)
        return torch.ones(20, device=device)
    return signal


def _open_writer(path, append):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if append else "w"
    handle = open(path, mode, newline="", encoding="utf-8")
    fieldnames = [
        "run",
        "elapsed_sec",
        "model",
        "n1",
        "n2",
        "sample_dim",
        "variant",
        "l0_lam",
        "l0_stat",
        "l0_thresh",
        "l0_decision",
        "l0_rej_rate",
        "l1_lam",
        "l1_stat",
        "l1_thresh",
        "l1_decision",
        "l1_rej_rate",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not append or not file_exists:
        writer.writeheader()
    return handle, writer


def run_l0(data1, data2, args, state):
    n1 = data1.size(0)
    n2 = data2.size(0)
    p1 = torch.ones(n1, device=device) / n1
    p2 = torch.ones(n2, device=device) / n2
    scale = (n1 * n2 / (n1 + n2)) ** 0.5

    if args.fixed_l0_lam is not None:
        state["l0_lam"] = torch.tensor(args.fixed_l0_lam, dtype=torch.float32, device=device)
    elif _should_tune(state["run"], args.tune_every, state.get("l0_lam"), args.fixed_l0_lam):
        state["l0_lam"] = tune_l0(
            data1,
            data2,
            state["l0_lam_seq"],
            k=args.cv_folds,
            reps=args.l0_tune_reps,
            n_l1=args.l0_l1_steps,
        )

    lam_l0 = state["l0_lam"]
    lam_l1 = _log_grid(lam_l0, args.l0_l1_steps)
    mswd_l0, _, _ = maxSlicedWDL0_L1Approx(
        data1,
        data2,
        p1,
        p2,
        lam_l0,
        lam_l1,
        candidate_adaptive=bool(args.candidate_adaptive),
        n_l1=args.l0_l1_steps,
        reps=args.reps,
    )
    stat = scale * mswd_l0
    thresh = None
    decision = None
    if args.bootstrap:
        thresh, _ = maxSlicedWDL0_L1Approx_bootstrap(
            data1,
            data2,
            lam_l0,
            lam_l1,
            candidate_adaptive=bool(args.candidate_adaptive),
            n_l1=args.l0_l1_steps,
            reps=args.l0_bootstrap_reps,
            B=args.l0_B,
            alpha=args.alpha,
        )
        decision = stat > thresh

    return {
        "lam": lam_l0,
        "stat": stat,
        "thresh": thresh,
        "decision": decision,
    }


def run_l1(data1, data2, args, state):
    n1 = data1.size(0)
    n2 = data2.size(0)
    sample_dim = data1.size(1)
    p1 = torch.ones(n1, device=device) / n1
    p2 = torch.ones(n2, device=device) / n2
    scale = (n1 * n2 / (n1 + n2)) ** 0.5

    if args.fixed_l1_lam is not None:
        state["l1_lam"] = torch.tensor(args.fixed_l1_lam, dtype=torch.float32, device=device)
    elif _should_tune(state["run"], args.tune_every, state.get("l1_lam"), args.fixed_l1_lam):
        state["l1_lam"] = tune_l1(
            data1,
            data2,
            state["l1_lam_seq"],
            k=args.cv_folds,
            reps=args.l1_tune_reps,
        )

    lam = state["l1_lam"]
    max_mswd = -1.0
    best_v0 = None
    best_v = None
    for _ in range(args.reps):
        v0 = torch.randn(sample_dim, 1, device=device)
        v0 = v0 / torch.norm(v0)
        v0 = myProj.myProj(v0, lam)
        mswd, v = maxSlicedWD(data1, data2, v0, p1, p2, lam=lam, learn_rate=100, thresh=1e-6)
        mswd_value = float(to_scalar(mswd))
        if mswd_value > max_mswd:
            max_mswd = mswd_value
            best_v0 = v0.clone()
            best_v = v.clone()

    stat = scale * max_mswd
    thresh = None
    decision = None
    if args.bootstrap:
        thresh, _ = maxSlicedWD_bootstrap(
            data1,
            data2,
            best_v0,
            lam=lam,
            B=args.nB,
            alpha=args.alpha,
            learn_rate=100,
            thresh=1e-6,
        )
        decision = stat > thresh

    return {
        "lam": lam,
        "stat": stat,
        "thresh": thresh,
        "decision": decision,
        "projection": best_v,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Accelerated MSWD-only simulation runner.")
    parser.add_argument("--model", default="mean-decay", type=str)
    parser.add_argument("--nrun", default=100, type=int)
    parser.add_argument("--n1", default=250, type=int)
    parser.add_argument("--n2", default=250, type=int)
    parser.add_argument("--sample_dim", default=500, type=int)
    parser.add_argument("--alpha", default=0.05, type=float)
    parser.add_argument("--variant", default="l0", choices=["l0", "l1", "both"])
    parser.add_argument("--signal", type=float, default=[1.0], nargs="+")
    parser.add_argument("--sig", type=str, default="")
    parser.add_argument("--seed", default=None, type=int)

    parser.add_argument("--bootstrap", default=1, type=int)
    parser.add_argument("--nB", default=200, type=int, help="L1 bootstrap samples.")
    parser.add_argument("--l0_B", default=150, type=int, help="L0 bootstrap samples.")
    parser.add_argument("--reps", default=5, type=int, help="Optimization restarts.")
    parser.add_argument("--l0_tune_reps", default=2, type=int)
    parser.add_argument("--l1_tune_reps", default=2, type=int)
    parser.add_argument("--l0_bootstrap_reps", default=1, type=int)
    parser.add_argument("--l0_l1_steps", default=6, type=int)
    parser.add_argument("--cv_folds", default=2, type=int)
    parser.add_argument("--candidate_adaptive", default=1, type=int)

    parser.add_argument(
        "--tune_every",
        default=0,
        type=int,
        help="0 tunes once and reuses lambdas; 1 tunes every run; N tunes every N runs.",
    )
    parser.add_argument("--fixed_l0_lam", default=None, type=float)
    parser.add_argument("--fixed_l1_lam", default=None, type=float)
    parser.add_argument("--out", default="results/mswd_acc_results.csv", type=str)
    parser.add_argument("--append", default=0, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    args.bootstrap = bool(args.bootstrap)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    signal = _prepare_signal(args)
    state = {
        "run": 0,
        "l0_lam": None,
        "l1_lam": None,
        "l0_lam_seq": torch.tensor([1, 5, 10, 20, 50], dtype=torch.float32, device=device),
        "l1_lam_seq": torch.exp(torch.linspace(np.log(1.5), np.log(5), steps=5, device=device)),
    }
    l0_decisions = []
    l1_decisions = []

    print(f"device {device}", flush=True)
    if device.type == "cuda":
        print(f"gpu {torch.cuda.get_device_name(device)}", flush=True)
    print(
        "accelerated settings: "
        f"variant={args.variant}, nrun={args.nrun}, tune_every={args.tune_every}, "
        f"bootstrap={int(args.bootstrap)}, l0_B={args.l0_B}, nB={args.nB}, reps={args.reps}",
        flush=True,
    )
    print(f"writing results to {args.out}", flush=True)

    handle, writer = _open_writer(args.out, bool(args.append))
    try:
        for run_id in range(args.nrun):
            state["run"] = run_id
            start = time.perf_counter()
            with torch.no_grad():
                data1, data2 = dataGenerate(args.n1, args.n2, args.sample_dim, signal, args.model)
                data1 = data1.to(device)
                data2 = data2.to(device)

                l0 = None
                l1 = None
                if args.variant in ("l0", "both"):
                    l0 = run_l0(data1, data2, args, state)
                    if l0["decision"] is not None:
                        l0_decisions.append(_as_int(l0["decision"]))
                if args.variant in ("l1", "both"):
                    l1 = run_l1(data1, data2, args, state)
                    if l1["decision"] is not None:
                        l1_decisions.append(_as_int(l1["decision"]))

            elapsed = time.perf_counter() - start
            l0_rate = float(np.mean(l0_decisions)) if l0_decisions else ""
            l1_rate = float(np.mean(l1_decisions)) if l1_decisions else ""
            row = {
                "run": run_id,
                "elapsed_sec": round(elapsed, 3),
                "model": args.model,
                "n1": args.n1,
                "n2": args.n2,
                "sample_dim": args.sample_dim,
                "variant": args.variant,
                "l0_lam": _as_float(l0["lam"]) if l0 else "",
                "l0_stat": _as_float(l0["stat"]) if l0 else "",
                "l0_thresh": _as_float(l0["thresh"]) if l0 else "",
                "l0_decision": _as_int(l0["decision"]) if l0 and l0["decision"] is not None else "",
                "l0_rej_rate": l0_rate,
                "l1_lam": _as_float(l1["lam"]) if l1 else "",
                "l1_stat": _as_float(l1["stat"]) if l1 else "",
                "l1_thresh": _as_float(l1["thresh"]) if l1 else "",
                "l1_decision": _as_int(l1["decision"]) if l1 and l1["decision"] is not None else "",
                "l1_rej_rate": l1_rate,
            }
            writer.writerow(row)
            handle.flush()
            print(
                f"run {run_id}: elapsed={elapsed:.1f}s, "
                f"l0_rej={l0_rate if l0_rate != '' else 'NA'}, "
                f"l1_rej={l1_rate if l1_rate != '' else 'NA'}",
                flush=True,
            )
    finally:
        handle.close()


if __name__ == "__main__":
    main()
