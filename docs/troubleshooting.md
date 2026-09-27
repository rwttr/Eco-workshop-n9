# Troubleshooting

[← README](../README.md) · [self-study guide](self-study-guide.md) · [trace analysis](trace-analysis.md)

## Both notebooks

| Symptom | Cause / fix |
|---|---|
| `USDT: ... SyncActivityProfilerHandler.cpp:39] profiler_start` on macOS | libtorch prints this to stderr whenever a profiler starts or stops. No supported way to silence it; harmless. |
| `Memory block of unknown size was allocated before the profiling started` | Expected: a tensor allocated before profiling was freed inside it. Doesn't affect conclusions. |
| Mermaid diagrams show as raw code | Viewer doesn't support Mermaid (older JupyterLab, nbviewer). Use JupyterLab 4.1+, VS Code or GitHub, or paste into <https://mermaid.live>. |
| Markdown text larger than the code | Run the Part 0 styling cell. Cosmetic only. |
| MPS unavailable | MPS cells check `torch.backends.mps.is_available()` and skip on Intel Macs, Linux and Windows. |
| Generated files | `profiler_out/`, `benchmark_out/` and `slow_data.py` are safe to delete. |

## Notebook 1 — profiler

| Symptom | Cause / fix |
|---|---|
| DataLoader workers hang or error | On macOS/Windows workers are spawned, so the `Dataset` must live in an importable module; that is why §3.4 writes `slow_data.py` with `%%writefile`. If it still fails, set `num_workers=0`. The section's point still holds. |
| `torch.compile` cell is slow | Off by default (`RUN_COMPILE = False`): ~30 s compile on a laptop CPU for ~20% gain here. |
| `export_stacks()` writes a zero-byte file | `with_stack=True` also needs `experimental_config=_ExperimentalConfig(verbose=True)`. Shown on purpose in §1.6. |

## Notebook 2 — benchmark

| Symptom | Cause / fix |
|---|---|
| Numbers differ a lot from the docs | Expected; figures are machine-specific. Direction and rough size should match. If not, check `torch.get_num_threads()` and background CPU load. |
| Benchmark cells are slow | By design: `blocked_autorange(min_run_time=t)` samples for `t` seconds. Lower `min_run_time` for speed, at the cost of wider error bars. |
| `Timer(language="c++")` fails | Needs `pip install ninja`, a C++ toolchain (`xcode-select --install` on macOS), and `ninja` on `PATH` (true if Jupyter was launched from the activated env). Or set `RUN_CPP = False`. |
| `collect_callgrind`: `Valgrind is not supported on this platform` | Expected off Linux. The cell catches it and explains what it would show. |

## Trace analysis

See [trace-analysis.md](trace-analysis.md) for context.

| Symptom | Cause / fix |
|---|---|
| HTA: `AttributeError: Can only use .str accessor with string values` | Trace has no `ProfilerStep#N` markers. Capture with a `schedule` + `tensorboard_trace_handler`, or use `ws_trace.capture_trace()`. |
| HTA prints many `Parsed ... / leaving parse_traces ...` lines | HTA logs at WARNING, so `logging.disable(logging.INFO)` doesn't help. Use `ws_trace.quiet_hta()`. |
| HTA: "If the trace file does not have the rank specified in it..." | Harmless; defaults to rank 0. `ws_trace.add_rank()` suppresses it. |
| HTA GPU analyses raise `IndexError` or `ValueError: No objects to concatenate` | Expected on a CPU-only trace: no GPU kernels to analyse. |
| `make_screenshots.py` fails or hangs | It drives headless Chromium against the live ui.perfetto.dev, so it needs network access and `python -m playwright install chromium`. Only needed to regenerate `docs/` screenshots. |
