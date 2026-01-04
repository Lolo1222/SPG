#!/usr/bin/env python3
"""
Wrapper entrypoint for accelerate launcher.
This script runs the training module `spg.diffu_grpo_train` using runpy.run_module,
so the module is executed with proper package context and relative imports work.
"""
import runpy
import sys
from pathlib import Path

def _ensure_repo_on_path():
    # Repo root is two parents up from this file (project root)
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

def _clean_accelerate_args(argv):
    # Known flags injected by accelerate/torchrun that should NOT be forwarded
    skip_flags = {
        "--num_processes",
        "--num_machines",
        "--main_process_port",
        "--mixed_precision",
        "--dynamo_backend",
        "--config_file",
        "--config",
    }
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in skip_flags:
            i += 1
            # skip a following value if it's not another flag
            if i < len(argv) and not str(argv[i]).startswith("--"):
                i += 1
            continue
        # remove common launcher-injected args or their values
        if a.startswith("--rdzv_") or a.startswith("--local_rank") or a.startswith("--node_rank"):
            i += 1
            # optional value
            if i < len(argv) and not str(argv[i]).startswith("--"):
                i += 1
            continue
        cleaned.append(a)
        i += 1
    return cleaned

if __name__ == "__main__":
    # Make local repo importable
    # _ensure_repo_on_path()

    # # Clean argv (but keep user flags such as --wait_for_debug)
    # sys.argv = _clean_accelerate_args(sys.argv)    
    # Forward any argv to the module via sys.argv
    # runpy.run_module will set sys.argv[0] to the module name when run_name='__main__'
    # We simply delegate execution to the module so it behaves like `python -m spg.diffu_grpo_train`.
    runpy.run_module("spg.diffu_grpo_train", run_name="__main__", alter_sys=True)
