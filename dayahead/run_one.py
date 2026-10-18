"""Run one experiment in its own process: python run_one.py <name> <period>.

PyTorch and LightGBM ship separate OpenMP runtimes that crash or deadlock when
both are used in one process, so each experiment gets a fresh process.
"""
import sys

name, period = sys.argv[1:3]
if name.startswith("mlp"):
    import torch  # noqa: F401  load torch before LightGBM, or it segfaults
import exp  # noqa: E402

exp.bt.LGB_PARAMS["num_threads"] = 3     # three experiments run side by side
exp.run(name, period)
