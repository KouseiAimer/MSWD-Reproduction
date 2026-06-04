"""Checkpointed outer runner for the original ../main.py.

Important: this script only passes --signal to main.py. All other numerical
settings inside main.py, including its default nrun, are left unchanged.

Each saved row is one complete execution of:
    python main.py --signal <0.8 * beta>

The outer runner can repeat that full main.py execution several times per beta
and resume if interrupted between executions.
"""

from __future__ import annotations

import argparse
import csv
import io
import runpy
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / "main.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BETA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CSV_COLUMNS = [
    "model",
    "beta",
    "signal",
    "main_run_id",
    "seed",
    "main_default_nrun",
    "n1",
    "n2",
    "sample_dim",
    "alpha",
    "proposed_rate",
    "mmd_g_rate",
    "mmd_l_rate",
    "ed2_rate",
    "bg_rate",
    "pw_rate",
    "stdout_log",
    "runtime_seconds",
    "started_at",
    "finished_at",
]


def parse_betas(raw: list[str] | None) -> list[float]:
    if not raw:
        return BETA_GRID
    if len(raw) == 1 and raw[0].lower() == "all":
        return BETA_GRID
    return [float(value) for value in raw]


def beta_key(beta: float) -> str:
    return f"{beta:.10g}"


def safe_beta_name(beta: float) -> str:
    return beta_key(beta).replace(".", "p").replace("-", "m")


def tensor_bool_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(bool(value.detach().to("cpu").item()))
    return int(bool(value))


def mean_from_global(globals_dict: dict[str, Any], name: str) -> float | str:
    value = globals_dict.get(name)
    if value is None:
        return ""
    arr = np.asarray(value, dtype=float)
    return float(np.mean(arr))


def read_completed_runs(path: Path) -> dict[str, set[int]]:
    completed: dict[str, set[int]] = {}
    if not path.exists():
        return completed
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row and row.get("proposed_rate", "") != "":
                completed.setdefault(beta_key(float(row["beta"])), set()).add(int(row["main_run_id"]))
    return completed


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in CSV_COLUMNS})
        f.flush()


def write_log(log_dir: Path, beta: float, main_run_id: int, text: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"beta_{safe_beta_name(beta)}_mainrun_{main_run_id:03d}.log"
    path.write_text(text, encoding="utf-8")
    return path


def execute_main_once(args: argparse.Namespace, beta: float, main_run_id: int) -> tuple[dict[str, Any], str]:
    """Run original main.py once, passing only --signal."""
    seed = args.seed_base + int(round(beta * 1000)) * 100000 + main_run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    signal = 0.8 * beta

    import pyCode.sim as sim_module

    orig_sim = sim_module.sim
    sim_results: list[dict[str, Any]] = []

    def sim_wrapper(data1: torch.Tensor, data2: torch.Tensor, opts: Any) -> Any:
        result = orig_sim(data1, data2, opts)
        sim_results.append(result)
        return result

    old_argv = sys.argv[:]
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    started_at = datetime.now().isoformat(timespec="seconds")
    start_time = time.time()

    try:
        sim_module.sim = sim_wrapper
        sys.argv = [str(MAIN_PATH), "--signal", str(signal)]
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            main_globals = runpy.run_path(str(MAIN_PATH), run_name="__main__")
    finally:
        sys.argv = old_argv
        sim_module.sim = orig_sim

    finished_at = datetime.now().isoformat(timespec="seconds")
    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    full_log = stdout_text
    if stderr_text:
        full_log += "\n\n[stderr]\n" + stderr_text

    opts = main_globals.get("opts")
    nrun = getattr(opts, "nrun", "")
    n1 = getattr(opts, "n1", "")
    n2 = getattr(opts, "n2", "")
    sample_dim = getattr(opts, "sample_dim", "")
    alpha = getattr(opts, "alpha", "")

    mmd_l_values = []
    for result in sim_results:
        perm = result.get("perm", {}) if isinstance(result, dict) else {}
        if "lapmmd_decision" in perm:
            mmd_l_values.append(tensor_bool_int(perm["lapmmd_decision"]))

    row: dict[str, Any] = {
        "model": "A_mean_decay",
        "beta": beta,
        "signal": signal,
        "main_run_id": main_run_id,
        "seed": seed,
        "main_default_nrun": nrun,
        "n1": n1,
        "n2": n2,
        "sample_dim": sample_dim,
        "alpha": alpha,
        "proposed_rate": mean_from_global(main_globals, "mswd_l0_decision"),
        "mmd_g_rate": mean_from_global(main_globals, "mmd_decision"),
        "mmd_l_rate": float(np.mean(mmd_l_values)) if mmd_l_values else "",
        "ed2_rate": mean_from_global(main_globals, "edl2_decision"),
        "bg_rate": mean_from_global(main_globals, "bg_decision"),
        "pw_rate": mean_from_global(main_globals, "kPWD_perm_decision"),
        "runtime_seconds": time.time() - start_time,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    return row, full_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeat original main.py with different signal values.")
    parser.add_argument("--beta", nargs="+", default=None, help="Beta value(s), e.g. --beta 0.2 or --beta all.")
    parser.add_argument("--main-runs", default=10, type=int, help="Number of complete main.py executions per beta.")
    parser.add_argument("--seed-base", default=20260602, type=int)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results" / "model_A_main_outputs.csv")
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    betas = parse_betas(args.beta)
    completed = read_completed_runs(args.out)

    print("Outer runner for original main.py")
    print(f"main.py: {MAIN_PATH}")
    print("main.py arguments passed by this runner: --signal only")
    print(f"raw output file: {args.out}")
    print(f"log dir: {args.log_dir}")
    print(f"beta values: {betas}")
    print(f"target complete main.py executions per beta: {args.main_runs}")

    new_runs = 0
    for beta in betas:
        key = beta_key(beta)
        done = completed.get(key, set())
        if len(done) >= args.main_runs:
            print(f"beta={beta}: already completed {len(done)}/{args.main_runs}; no more main.py executions needed.")
            continue

        print(f"beta={beta}: completed {len(done)}/{args.main_runs}; continuing.")
        for main_run_id in range(1, args.main_runs + 1):
            if main_run_id in done:
                continue

            print(f"beta={beta}, main_run={main_run_id}/{args.main_runs}: executing main.py --signal {0.8 * beta}")
            row, log_text = execute_main_once(args, beta, main_run_id)
            log_path = write_log(args.log_dir, beta, main_run_id, log_text)
            row["stdout_log"] = str(log_path)
            append_result(args.out, row)
            done.add(main_run_id)
            completed.setdefault(key, set()).add(main_run_id)
            new_runs += 1
            print(
                f"beta={beta}, main_run={main_run_id}/{args.main_runs}: saved; "
                f"Proposed={row['proposed_rate']}, MMD-G={row['mmd_g_rate']}, "
                f"MMD-L={row['mmd_l_rate']}, ED2={row['ed2_rate']}, "
                f"BG={row['bg_rate']}, PW={row['pw_rate']}; log={log_path}"
            )

        print(f"beta={beta}: completed {len(done)}/{args.main_runs}.")

    if new_runs == 0:
        print("No new main.py executions were needed.")


if __name__ == "__main__":
    main()
