# Troubleshooting

[← back to README](../README.md) · [self-study guide](self-study-guide.md)

---

## Both notebooks

**`USDT: ... SyncActivityProfilerHandler.cpp:39] profiler_start` messages on macOS**
These originate in libtorch's tracing backend and are written to stderr whenever a profiler
starts or stops. No supported environment variable suppresses them, and they can be safely
ignored.

**`Memory block of unknown size was allocated before the profiling started`**
Expected. A tensor allocated before the profiler started was freed within the profiled
region, so that deallocation cannot be attributed. It does not affect the conclusions.

**Mermaid diagrams appear as raw code blocks**
The viewer does not render Mermaid. JupyterLab 4.1+, VS Code and GitHub do; older versions
of JupyterLab and nbviewer do not. The block can be pasted into <https://mermaid.live>.

**Markdown text remains larger than the code**
Part 0 injects a stylesheet that reduces markdown prose, headings and tables to
approximately the code font size. It is an ordinary code cell; if it was skipped, run it.
The change affects appearance only.

**MPS unavailable**
Every MPS cell is guarded by `torch.backends.mps.is_available()` and is skipped cleanly on
Intel Macs, Linux and Windows.

**Generated files**
Running the notebooks creates `profiler_out/`, `benchmark_out/` and `slow_data.py`. All can
be deleted safely.

---

## Notebook 1 — profiler

**DataLoader workers hang or raise an error**
Part 3.4 writes `slow_data.py` from a notebook cell using `%%writefile` deliberately. On
macOS and Windows, worker processes are spawned, so the `Dataset` class must reside in an
importable module; a class defined in a notebook cell cannot be pickled and sent to a
worker. If problems persist, set `num_workers=0`; the remainder of the section still
demonstrates its point.

**The `torch.compile` cell is slow**
It is disabled by default (`RUN_COMPILE = False`). Compilation takes approximately 30
seconds on a laptop CPU for roughly a 20% improvement on this model. The notebook states
this trade-off explicitly rather than concealing it.

**`export_stacks()` writes a zero-byte file**
`with_stack=True` is not sufficient on its own. It must be accompanied by
`experimental_config=_ExperimentalConfig(verbose=True)`. This behaviour is demonstrated
deliberately in §1.6.

---

## Notebook 2 — benchmark

**Benchmark results differ substantially from those in the documentation**
Expected, as the figures are machine-specific. What should reproduce is the direction and
approximate magnitude of each effect. If those also disagree, check
`torch.get_num_threads()` and whether other processes are using the CPU.

**Benchmark cells take a long time**
This is by design: `blocked_autorange(min_run_time=t)` continues sampling for `t` seconds
of measurement. Reducing `min_run_time` shortens the notebook at the cost of wider error
bars.

**`Timer(language="c++")` fails**
This requires `pip install ninja` together with a C++ toolchain (`xcode-select --install`
on macOS), and `ninja` must be present on `PATH`, which is the case when Jupyter is
launched from the activated environment. Set `RUN_CPP = False` to skip the cell.

**`collect_callgrind` raises `Valgrind is not supported on this platform`**
Expected on macOS, as Valgrind is available only on Linux. The cell catches the exception
and describes what the analysis would provide on a supported platform.

---

## Trace analysis

Further context for each of the following is given in
[trace-analysis.md](trace-analysis.md).

**HTA raises `AttributeError: Can only use .str accessor with string values`**
The trace contains no `ProfilerStep#N` markers. Capture it using a `schedule` together with
`tensorboard_trace_handler`, or use `ws_trace.capture_trace()`.

**HTA prints a long sequence of `Parsed ... / leaving parse_traces ...` lines**
`logging.disable(logging.INFO)` does not suppress these, as HTA logs at WARNING level. Use
`ws_trace.quiet_hta()`, which sets `logging.getLogger("hta").setLevel(logging.ERROR)`.

**HTA reports "If the trace file does not have the rank specified in it..."**
Harmless; it defaults to rank 0. `ws_trace.add_rank()` writes `distributedInfo` into the
file to suppress the message.

**HTA GPU analyses raise `IndexError` or `ValueError: No objects to concatenate`**
Expected on a CPU-only trace, which contains no GPU kernels to analyse.

**`make_screenshots.py` fails or does not complete**
The script drives a headless Chromium instance against the live ui.perfetto.dev, so it
requires network access and `python -m playwright install chromium`. It is needed only to
regenerate the screenshots in `docs/`; the notebooks do not depend on it.
