"""Trace capture and analysis helpers for the 241-353 AI Ecosystem workshops.

Two ways to look at a PyTorch profiler trace:

1. **Chrome Trace / Perfetto** - the simple path. One JSON file, drag it into a browser,
   or parse it with `json` + `pandas` using `chrome_trace_ops()` below. No extra
   dependencies beyond what you already have.

2. **Holistic Trace Analysis (HTA)** - Meta's trace analysis library. Built for large
   multi-GPU distributed jobs; gives you a normalised event DataFrame, profiler-step
   detection and trace diffing. `pip install HolisticTraceAnalysis`.

The one non-obvious requirement for HTA: **it needs `ProfilerStep#N` markers**, which only
appear when you profile with a `schedule` AND write the file with
`tensorboard_trace_handler`. A manual `prof.export_chrome_trace(...)` does not emit them,
and HTA fails to parse the result. `capture_trace()` does it the right way.
"""

import json
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

__all__ = ["capture_trace", "add_rank", "chrome_trace_events", "chrome_trace_ops",
           "hta_available", "quiet_hta", "HTA_ON_CPU"]


def quiet_hta(level=None):
    """Silence HTA's per-file parsing chatter.

    HTA logs "Parsed ... / leaving parse_traces ..." through a logger named `hta`, at
    **WARNING** level - so the obvious `logging.disable(logging.INFO)` does nothing at all.
    You have to raise that specific logger's level instead.
    """
    import logging
    logging.getLogger("hta").setLevel(level or logging.ERROR)


# --------------------------------------------------------------------- capture
def add_rank(trace_file, rank=0):
    """Stamp a distributed rank into a trace file.

    HTA identifies ranks from `distributedInfo`. A single-process trace has none, so HTA
    warns and assumes rank 0. Writing it explicitly silences the warning and is required
    if you ever compare several processes.
    """
    trace_file = Path(trace_file)
    data = json.loads(trace_file.read_text())
    data["distributedInfo"] = {"rank": rank}
    trace_file.write_text(json.dumps(data))
    return trace_file


def capture_trace(fn, base_dir, name, wait=0, warmup=1, active=3,
                  record_shapes=True, profile_memory=False, with_stack=False,
                  experimental_config=None, rank=0, activities=None):
    """Run `fn` under a scheduled profiler and write one HTA-compatible trace.

    Each trace goes in its own directory, because `TraceAnalysis(trace_dir=...)` loads
    *every* file in a directory and treats each as a different rank.

    Returns `(trace_dir, trace_file)`: pass the directory to `TraceAnalysis(trace_dir=...)`
    and the file to Perfetto or `chrome_trace_ops()`.
    """
    out = Path(base_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.json"):
        stale.unlink()

    n_steps = wait + warmup + active
    with profile(
        activities=activities or [ProfilerActivity.CPU],
        schedule=schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=tensorboard_trace_handler(str(out)),   # <- emits ProfilerStep#N
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(n_steps):
            fn()
            prof.step()

    files = sorted(out.glob("*.pt.trace.json"))
    if not files:
        raise RuntimeError(f"no trace written to {out}")
    return out, add_rank(files[-1], rank)


# ------------------------------------------------- the simple path: raw chrome trace
def chrome_trace_events(path):
    """Load the complete-duration events ('ph' == 'X') of a Chrome Trace into a DataFrame."""
    data = json.loads(Path(path).read_text())
    rows = [
        {"name": e["name"], "cat": e.get("cat", ""), "ts": e["ts"],
         "dur": e.get("dur", 0), "pid": e.get("pid"), "tid": e.get("tid")}
        for e in data["traceEvents"] if e.get("ph") == "X"
    ]
    return pd.DataFrame(rows)


def chrome_trace_ops(path, top=15, cat="cpu_op"):
    """Per-operator self and total time from a raw Chrome Trace, with no HTA involved.

    Durations in a Chrome Trace are *inclusive* (an op contains its children), so summing
    them double-counts nesting. We recover self time the same way the profiler does: sweep
    each thread's events in time order, keep a stack of open parents, and subtract each
    child's duration from its parent.
    """
    df = chrome_trace_events(path)
    if df.empty:
        return df
    df = df.sort_values(["pid", "tid", "ts"], kind="stable").reset_index(drop=True)
    self_dur = df["dur"].astype(float).to_numpy().copy()

    for _, grp in df.groupby(["pid", "tid"], sort=False):
        stack = []                                   # (end_ts, row_index)
        for idx, ts, dur in zip(grp.index, grp["ts"], grp["dur"]):
            end = ts + dur
            while stack and stack[-1][0] <= ts:      # close finished parents
                stack.pop()
            if stack:
                self_dur[stack[-1][1]] -= dur        # this event is a child
            stack.append((end, idx))

    df["self_dur"] = self_dur
    if cat is not None:
        df = df[df.cat == cat]
    out = (df.groupby("name")
             .agg(calls=("dur", "size"),
                  self_ms=("self_dur", lambda s: s.sum() / 1e3),
                  total_ms=("dur", lambda s: s.sum() / 1e3))
             .sort_values("self_ms", ascending=False))
    out["self_%"] = 100 * out.self_ms / out.self_ms.sum()
    return out.head(top).reset_index()


# ----------------------------------------------------------------------- HTA info
def hta_available():
    """True if HolisticTraceAnalysis is importable."""
    try:
        import hta  # noqa: F401
        return True
    except ImportError:
        return False


# What each HTA analysis needs. On a CPU-only trace the GPU ones have no data to work on.
HTA_ON_CPU = pd.DataFrame([
    ("get_profiler_steps",             "step boundaries",        "works on CPU"),
    ("t.get_trace(rank)",              "normalised event table", "works on CPU"),
    ("t.symbol_table",                 "name/category decoding", "works on CPU"),
    ("TraceDiff.compare_traces",       "op counts + durations, two traces", "works on CPU"),
    ("TraceDiff.ops_diff",             "added / removed / changed ops",     "works on CPU"),
    ("get_temporal_breakdown",         "compute vs comms vs idle, per rank", "needs GPU"),
    ("get_idle_time_breakdown",        "why the GPU was idle",   "needs GPU"),
    ("get_gpu_kernel_breakdown",       "top CUDA kernels",       "needs GPU"),
    ("get_cuda_kernel_launch_stats",   "launch overhead / runtime gaps",    "needs GPU"),
    ("get_comm_comp_overlap",          "all-reduce vs compute overlap",     "needs multi-GPU"),
    ("get_memory_bw_time_series",      "achieved memory bandwidth",         "needs GPU"),
    ("critical_path_analysis",         "longest dependency chain",          "needs GPU"),
], columns=["HTA call", "what it tells you", "on a CPU-only trace"])
