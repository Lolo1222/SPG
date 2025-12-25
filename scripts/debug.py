#!/usr/bin/env python3
"""
占用指定 GPU 指定显存的脚本。

用法示例:
  python scripts/gpu_mem_hog.py --gpus 0,1 --mem 2048 --duration 3600

支持：
 - 使用 PyTorch（优先）或 CuPy（备用）来分配显存。
 - 可以为每个 GPU 指定统一内存（--mem）或按 GPU 列表指定每卡内存（--mem-list）。
 - duration=0 表示直到 Ctrl+C。
"""
import argparse
import signal
import sys
import time
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    # 帮助静态类型检查器/Pylance 识别 cupy；运行时使用 importlib 动态导入
    import cupy as cp  # type: ignore


def parse_gpu_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def parse_mem_list(s: str) -> List[int]:
    if s is None or s.strip() == "":
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def try_import_cupy():
    try:
        import importlib

        cp = importlib.import_module("cupy")

        return cp
    except Exception:
        return None


def human_mb(n):
    return f"{n} MB"


def allocate_with_torch(torch, device: int, mb: int):
    # dtype uint8 -> 1 byte per element
    bytes_to_alloc = mb * 1024 * 1024
    num_elems = bytes_to_alloc
    dev_str = f"cuda:{device}"
    torch.cuda.set_device(device)
    print(f"Allocating {human_mb(mb)} on {dev_str} via PyTorch...")
    # allocate a 1D tensor of uint8
    t = torch.empty(num_elems, dtype=torch.uint8, device=dev_str)
    # touch the tensor to ensure allocation
    t[0] = 0
    t[-1] = 0
    return t


def allocate_with_cupy(cp, device: int, mb: int):
    bytes_to_alloc = mb * 1024 * 1024
    num_elems = bytes_to_alloc
    print(f"Allocating {human_mb(mb)} on GPU {device} via CuPy...")
    with cp.cuda.Device(device):
        a = cp.empty(num_elems, dtype=cp.uint8)
        a[0] = 0
        a[-1] = 0
    return a


def main():
    parser = argparse.ArgumentParser(description="占用指定 GPU 的显存")
    parser.add_argument("--gpus", default="0", help="逗号分隔的 GPU id 列表，例如 0,1")
    parser.add_argument("--mem", type=int, default=1024, help="每张卡占用显存 (MB)，与 --mem-list 二选一")
    parser.add_argument("--mem-list", default=None, help="按 GPU 列表的逐卡显存，逗号分隔 (MB)，例如 1024,2048")
    parser.add_argument("--duration", type=int, default=0, help="持续时间秒，0 表示直到 Ctrl+C")
    args = parser.parse_args()

    gpus = parse_gpu_list(args.gpus)
    if not gpus:
        print("未指定 GPU 列表，退出。")
        sys.exit(1)

    mem_list = None
    if args.mem_list:
        mem_list = parse_mem_list(args.mem_list)
        if len(mem_list) != len(gpus):
            print("当使用 --mem-list 时，长度必须与 --gpus 中的 GPU 数量一致。")
            sys.exit(1)

    torch = try_import_torch()
    cp = None
    if torch is None:
        cp = try_import_cupy()

    if torch is None and cp is None:
        print("既没有检测到 PyTorch 也没有检测到 CuPy。请安装其中之一以使用该脚本。")
        sys.exit(1)

    # track allocations so GC 不会释放
    allocations = []

    def cleanup(signum=None, frame=None):
        print("\n收到退出信号，释放显存并退出...")
        allocations.clear()
        # 在 PyTorch 中建议调用 empty_cache
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for i, dev in enumerate(gpus):
        mb = mem_list[i] if mem_list is not None else args.mem
        try:
            if torch is not None:
                if not torch.cuda.is_available():
                    print("PyTorch 已安装，但未检测到可用 CUDA。")
                    sys.exit(1)
                if dev >= torch.cuda.device_count():
                    print(f"GPU {dev} 不存在 (PyTorch 可见设备数={torch.cuda.device_count()})")
                    sys.exit(1)
                t = allocate_with_torch(torch, dev, mb)
                allocations.append(t)
            else:
                # use cupy
                if dev >= cp.cuda.runtime.getDeviceCount():
                    print(f"GPU {dev} 不存在 (CuPy 可见设备数={cp.cuda.runtime.getDeviceCount()})")
                    sys.exit(1)
                a = allocate_with_cupy(cp, dev, mb)
                allocations.append(a)
        except Exception as e:
            print(f"在 GPU {dev} 上分配失败: {e}")
            cleanup()

    print("全部分配完成。按 Ctrl+C 释放显存并退出。")
    if args.duration > 0:
        try:
            t0 = time.time()
            while True:
                elapsed = time.time() - t0
                if elapsed >= args.duration:
                    break
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        cleanup()
    else:
        # wait until signal
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            cleanup()


if __name__ == "__main__":
    main()
