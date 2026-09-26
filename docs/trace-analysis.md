# Trace analysis: Perfetto, HTA and manual parsing

[← back to README](../README.md) · [self-study guide](self-study-guide.md) · [models](models.md) · [profiler notebook](profiler-notebook.md) · [benchmark notebook](benchmark-notebook.md)

Both notebooks conclude by writing traces and analysing them. Three approaches are
available, in increasing order of cost.

| Approach | Requires | Suited to |
|---|---|---|
| **Perfetto** or `chrome://tracing` | a browser | Visual inspection: gaps, stalls, ordering, operator density |
| **Parsing the JSON directly** | `json` and `pandas` | Scripted checks, CI, custom metrics |
| **Holistic Trace Analysis** | `pip install HolisticTraceAnalysis` | Multi-rank GPU jobs, trace diffing |

---

## Viewer output

**Perfetto.** The trace from `pytorch_profiler_workshop.ipynb` Part 5, loaded at
<https://ui.perfetto.dev>. The three `ProfilerStep#N` spans correspond to the `active`
steps of the profiler `schedule`; `## train_step ##` beneath them is the `record_function`
label defined in the notebook, and the remaining rows show the operator nesting.

![PyTorch trace in the Perfetto UI](perfetto_overview.png)

Zooming in (**W** to zoom, **A** and **D** to pan) makes the individual operators legible.
Hovering over a slice reports its exact duration; the example below shows
`ConvolutionBackward0` at 2.891 ms.

![Perfetto zoomed into one ProfilerStep](perfetto_zoom.png)

**Holistic Trace Analysis** is a library rather than a graphical application, so it is used
from the notebook. `TraceAnalysis` parses the trace into a DataFrame, the symbol table
decodes the integer operator identifiers back into names, and `TraceDiff.ops_diff` reports
the differences between two traces. In the example below it correctly identifies the
`aten::relu` and `aten::add` regression introduced deliberately for the demonstration.

![Holistic Trace Analysis output](hta_analysis.png)

**Manual parsing.** The same trace processed with `json` and `pandas` alone. The second
table is a verification step: call counts are identical and self times agree with
`prof.key_averages()` to within run-to-run variation, using only the file the profiler had
already written.

![Chrome Trace parsed with pandas](chrome_trace_table.png)

> These screenshots are captured from live sessions. They can be regenerated after
> re-running the notebooks with `python make_screenshots.py`, which requires
> `pip install playwright && python -m playwright install chromium`.
> In current versions of Chrome, `chrome://tracing` redirects to the same Perfetto
> interface, so a single screenshot covers both.

---

## A requirement that is easy to overlook

**HTA requires `ProfilerStep#N` markers.** These are present only when the profiling run
uses a `schedule` *and* writes the file through `tensorboard_trace_handler`. A manual call
to `prof.export_chrome_trace(...)`, even from a scheduled profiler, does not include them,
and HTA then fails with a misleading message:

```
AttributeError: Can only use .str accessor with string values
```

This indicates that HTA found no ProfilerStep symbols; it is unrelated to the installed
version of pandas. `ws_trace.capture_trace()` performs the capture correctly and places
each trace in its own directory, because `TraceAnalysis(trace_dir=...)` loads every file in
a directory and treats each as a separate distributed rank.

## Suppressing HTA's logging output

HTA emits a line for every file it processes (`Parsed ... / leaving parse_traces ...`).
These are written through a logger named `hta` at **WARNING** level, so the conventional
`logging.disable(logging.INFO)` has no effect. The level of that specific logger must be
raised instead:

```python
logging.getLogger("hta").setLevel(logging.ERROR)     # equivalent to ws_trace.quiet_hta()
```

---

## Scope: HTA targets GPU workloads

Most of HTA's principal analyses read CUDA kernel and NCCL events. A CPU-only trace
contains neither, so those functions return empty results or raise an exception. Both
notebooks demonstrate this explicitly rather than omitting it.

| HTA call | Behaviour on a CPU-only trace |
|---|---|
| `get_profiler_steps()`, `t.get_trace(rank)`, `t.symbol_table` | **Supported** |
| `TraceDiff.compare_traces()`, `TraceDiff.ops_diff()` | **Supported**, and the principal reason to install the package |
| `get_temporal_breakdown()`, `get_idle_time_breakdown()` | Requires GPU |
| `get_gpu_kernel_breakdown()`, `get_cuda_kernel_launch_stats()` | Requires GPU |
| `get_comm_comp_overlap()`, `critical_path_analysis()` | Requires GPU or multiple GPUs |

On CPU, therefore, HTA is worth installing for `TraceDiff`, with Perfetto used for
everything else. On a single GPU it contributes substantially more, and on multi-GPU
workloads it has few alternatives. The capture code is identical in all three cases, so
writing traces in the HTA-compatible form from the outset keeps all options available.

## Interpreting a diff: use `diff_counts`, not `diff_duration`

Call counts in a trace diff are exact and deterministic. The duration columns are not: they
derive from a single profiled run, they include the profiler's own overhead, and HTA's
self-time approximation over a deeply nested CPU trace can produce negative values.
Determine *whether* something changed with `ab_compare()`, and *what* changed with
`diff_counts`.

---

## Parsing a trace directly

A Chrome Trace is a JSON document containing events with `name`, `ts`, `dur`, `cat`, `pid`
and `tid` fields. `ws_trace.chrome_trace_ops()` parses one using `pandas` alone and
reconstructs **self time** by traversing each thread in time order and subtracting each
child's duration from that of its parent. The profiler notebook verifies its output against
`prof.key_averages()`, and the two agree to within run-to-run variation. Consequently the
analysis is not tied to any single viewer, and a performance check can be added to CI
without introducing a dependency.

## Generated traces

Running the notebooks writes `profiler_out/traces/` and `benchmark_out/traces/`. Any
`.pt.trace.json` file can be opened by dragging it into <https://ui.perfetto.dev>; the
viewer runs locally in the browser and the file is not uploaded. Navigation uses **W** and
**S** to zoom, **A** and **D** to pan, a click to inspect an operator, and a drag to select
a range for aggregate statistics.

> Traces of operator-heavy models grow quickly. The naive Mamba scan issues approximately
> 27,000 events per forward pass, so three profiled steps produce a file of roughly 30 MB.
> The notebooks use `active=1` for those captures.

---

## References

- [Perfetto UI](https://ui.perfetto.dev), a trace viewer that runs locally in the browser
- [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) · [documentation](https://hta.readthedocs.io/)
- [Chrome Trace Event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview), the JSON schema underlying every trace used here
