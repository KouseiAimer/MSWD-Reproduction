"""GPU entry point for the MSWD simulation runner."""

import os

import torch


if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use main.py for CPU/auto device execution.")

os.environ["MSWD_DEVICE"] = "cuda"

from main import main  # noqa: E402


if __name__ == "__main__":
    main()
