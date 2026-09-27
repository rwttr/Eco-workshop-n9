# Notebook 2 — `pytorch_benchmark_workshop.ipynb`

`torch.utils.benchmark`: **how much** time a workload takes, and whether a change is real.

[← README](../README.md) · [self-study guide](self-study-guide.md) · [models](models.md) · [profiler notebook](profiler-notebook.md) · [trace analysis](trace-analysis.md)

## Contents

| Part | Topic | Time |
|---|---|---|
| **0** | Setup and shared model definitions | 5 min |
| **1** | Five ways `time.perf_counter()` misleads, each demonstrated | 20 min |
| **2** | `Timer`: `stmt`/`setup`/`globals`; `timeit` vs `blocked_autorange` vs `adaptive_autorange`; `num_threads` | 25 min |
| **3** | `Measurement`: median vs mean, IQR, `significant_figures`, a defensible A/B test | 25 min |
| **4** | `Compare`: nested sweeps as tables | 15 min |
| **5** | **Inference**: batch sweeps, autograd contexts, CPU vs MPS, regression gates | 30 min |
| **6** | **Training**: step breakdown, optimisers, `set_to_none`, epoch planning | 30 min |
| **7** | `Fuzzer`, C++ timing, Callgrind, benchmark → trace diff | 30 min |
| **8** | Combined workflow, checklist, exercises | 10 min |

## Features covered

- `Timer(stmt, setup, globals, label, sub_label, description, num_threads)`
- `timeit()`, `blocked_autorange()`, `adaptive_autorange()`, and when to use each
- `Measurement`: `.median`, `.mean`, `.iqr`, `.times`, `.number_per_run`,
  `.significant_figures`, `.meets_confidence()`, `.has_warnings`
- `Compare` with `trim_significant_figures()`, `colorize(rowwise=True)`, thread row-groups
- `Fuzzer`, `FuzzedParameter`, `FuzzedTensor`: random but reproducible shape sweeps
- `Timer(language="c++")` to isolate Python's per-operator overhead
- `collect_callgrind()` for deterministic instruction counts (Linux only; guarded)
- Two reusable helpers: `ab_compare()`, which refuses to report differences within noise, and
  `assert_not_slower()`, a regression gate that widens its tolerance on a noisy machine
- **Link to notebook 1**: trace the A and B variants, then diff them with HTA `TraceDiff` and
  a dependency-free Chrome Trace diff ([trace-analysis.md](trace-analysis.md))

## Findings expected to reproduce

1. **Unsynchronised accelerator timing is >200× too fast.** An MPS matmul timed without
   `synchronize()` reports ~0.03 ms for ~6 ms of work. `Timer` synchronises automatically
   via `torch.accelerator.synchronize()`.
2. **The null experiment.** Benchmarking the same work twice shows a 5–10% "change". That is
   the noise floor; claims below it mean nothing.
3. **Autograd context cost scales with operator density.** `inference_mode` vs gradients
   enabled: ~0% for ResNet-18, ~13% for the operator-heavy Mamba model.
4. **Interleave measurements; don't group them.** An earlier version of this cell ran each
   variant to completion in turn and got the order wrong, because machine drift exceeded the
   effect. Round-robin sampling fixes it. The notebook keeps both versions.
5. **`set_to_none=True` saves memory, not time.** It is ~5% faster, but the real gain is
   releasing ~43 MB of gradient buffers that `set_to_none=False` holds for the whole run.
6. **Benchmarks don't interpolate.** Per-sample training cost is not monotonic in batch size:
   batch 16 is ~60% worse per epoch than batch 8, with ~1% IQR, so the effect is real.
7. **MPS is not always faster.** ~4× faster than CPU on ResNet-18, >10× on MobileNetV3-Small
   (depthwise convs have no fast CPU path), but *slower* on Mamba, whose Python loop of tiny
   kernels leaves the GPU idle between launches.
8. **Python costs ~150 ns per operator.** `x + x` takes ~245 ns from C++ vs ~397 ns from
   Python. Times the 27,000 operator calls per Mamba forward pass, that is most of its runtime.

## References

- [`torch.utils.benchmark` API](https://docs.pytorch.org/docs/stable/benchmark_utils.html) · [Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [Valgrind / Callgrind manual](https://valgrind.org/docs/manual/cl-manual.html)
