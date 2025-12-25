#!/usr/bin/env python3
"""
Wrapper entrypoint for accelerate launcher.
This script runs the training module `spg.diffu_grpo_train` using runpy.run_module,
so the module is executed with proper package context and relative imports work.
"""
import runpy
import sys

if __name__ == "__main__":
    # Forward any argv to the module via sys.argv
    # runpy.run_module will set sys.argv[0] to the module name when run_name='__main__'
    # We simply delegate execution to the module so it behaves like `python -m spg.diffu_grpo_train`.
    runpy.run_module("spg.diffu_grpo_train", run_name="__main__", alter_sys=True)
