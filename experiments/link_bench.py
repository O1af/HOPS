"""Distributed link benchmark for fitting HOPS link overrides.

Run under torchrun on the target cluster. Output is JSON lines so results can be
redirected directly into a file and parsed later.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure distributed GPU link timings")
    parser.add_argument("--mode", choices=("p2p", "allreduce"), required=True)
    parser.add_argument("--sizes-mb", default="1,4,16,64,256")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--label", default="")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def bytes_per_element(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype is torch.float32:
        return 4
    raise ValueError(f"Unsupported dtype {dtype}")


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p50_index = len(ordered) // 2
    p99_index = max(0, min(len(ordered) - 1, int(round(0.99 * len(ordered))) - 1))
    return {
        "mean_ms": statistics.mean(ordered),
        "p50_ms": ordered[p50_index],
        "p99_ms": ordered[p99_index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "std_ms": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def timed_loop(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings


def main() -> None:
    args = parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dtype = torch_dtype(args.dtype)

    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    hostname = socket.gethostname()

    size_mbs = [int(item) for item in args.sizes_mb.split(",") if item]

    for size_mb in size_mbs:
        numel = (size_mb * 1024 * 1024) // bytes_per_element(dtype)
        tensor = torch.empty(numel, dtype=dtype, device="cuda").normal_()

        if args.mode == "p2p":
            if world_size != 2:
                raise ValueError("p2p mode requires world_size=2")
            peer = 1 - rank
            if rank == 0:
                op = lambda: dist.send(tensor, dst=peer)
            else:
                recv_buffer = torch.empty_like(tensor)
                op = lambda: dist.recv(recv_buffer, src=peer)
        else:
            op = lambda: dist.all_reduce(tensor)

        dist.barrier()
        timings = timed_loop(op, warmup=args.warmup, iters=args.iters)
        dist.barrier()

        row = {
            "label": args.label,
            "mode": args.mode,
            "rank": rank,
            "world_size": world_size,
            "hostname": hostname,
            "size_mb": size_mb,
            "dtype": args.dtype,
            "local_rank": local_rank,
            "warmup": args.warmup,
            "iters": args.iters,
        }
        row.update(summarize(timings))
        print(json.dumps(row), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
