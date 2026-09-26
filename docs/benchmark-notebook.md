# Notebook 2 — `pytorch_benchmark_workshop.ipynb`

Using `torch.utils.benchmark` to establish **how much** time a workload takes, and whether
an observed change is real.

[← back to README](../README.md) · [self-study guide](self-study-guide.md) · [models](models.md) · [profiler notebook](profiler-notebook.md) · [trace analysis](trace-analysis.md)

---

## Contents

| Part | Topic | Approx. time |
|---|---|---|
| **0** | Setup, styling and the shared model definitions | 5 min |
| **1** | The limitations of `time.perf_counter()`: five failure modes, each demonstrated | 20 min |
| **2** | `Timer`: `stmt`, `setup` and `globals`; `timeit` against `blocked_autorange` and `adaptive_autorange`; `num_threads` | 25 min |
| **3** | `Measurement`: median against mean, IQR, `significant_figures`, and a defensible A/B test | 25 min |
| **4** | `Compare`: presenting nested sweeps as tables | 15 min |
| **5** | **Benchmarking inference**: batch sweeps, autograd contexts, CPU against MPS, regression gates | 30 min |
| **6** | **Benchmarking training**: step breakdown, optimisers, `set_to_none`, epoch planning | 30 min |
| **7** | `Fuzzer`, C++ timing, Callgrind, and benchmark-to-trace-diff | 30 min |
| **8** | Combined workflow, checklist and exercises | 10 min |

---

## Features covered

- `Timer(stmt, setup, globals, label, sub_label, description, num_threads)`
- `timeit()`, `blocked_autorange()` and `adaptive_autorange()`, and the circumstances in
  which each is appropriate
- `Measurement`: `.median`, `.mean`, `.iqr`, `.times`, `.number_per_run`,
  `.significant_figures`, `.meets_confidence()` and `.has_warnings`
- `Compare` with `trim_significant_figures()`, `colorize(rowwise=True)` and thread
  row-groups
- `Fuzzer`, `FuzzedParameter` and `FuzzedTensor` for randomised but reproducible shape
  sweeps
- `Timer(language="c++")` for isolating Python's per-operator overhead
- `collect_callgrind()` for deterministic instruction counts, which requires Linux and is
  guarded and explained in the notebook
- Two reusable helpers developed in the notebook: `ab_compare()`, which declines to report
  differences within measurement noise, and `assert_not_slower()`, a regression gate that
  widens its tolerance on a noisy machine
- **Connecting the two notebooks**: capturing traces of the A and B variants, then applying
  HTA `TraceDiff` and a dependency-free Chrome Trace diff to identify which operators
  account for the difference. See [trace-analysis.md](trace-analysis.md).

---

## Findings expected to reproduce

1. **Unsynchronised accelerator timing is more than 200× too fast.** Timing an MPS
   matmul without `synchronize()` reports approximately 0.03 ms for work that takes
   approximately 6 ms. `Timer` synchronises automatically, since its default clock calls
   `torch.accelerator.synchronize()`.
2. **The null experiment.** Benchmarking the same work twice yields an apparent change of
   5–10%. This is the measurement noise floor, and no claim below it is meaningful.
3. **The cost of the autograd context depends on operator density.** The difference between
   `inference_mode` and gradient-enabled execution is close to zero for ResNet-18 and
   approximately 13% for the operator-heavy Mamba model. Per-operator overheads affect
   operator-heavy models most.
4. **Measurements should be interleaved rather than grouped.** An earlier version of that
   cell measured each variant to completion in turn and produced an incorrect ordering,
   because machine drift over the cell exceeded the effect being measured. Round-robin
   sampling resolves this. The notebook retains both versions and explains the difference.
5. **`set_to_none=True` is a memory optimisation rather than a speed optimisation.** It is
   approximately 5% faster, but it releases approximately 43 MB of gradient buffers that
   `set_to_none=False` retains for the duration of the run.
6. **Benchmark results cannot be interpolated.** Per-sample training cost is not monotonic
   in batch size; on the reference machine, batch 16 is approximately 60% worse per epoch
   than batch 8, with an IQR of approximately 1%, indicating a real effect rather than
   noise.
7. **MPS is not universally faster.** It outperforms the CPU by approximately 4× on
   ResNet-18 and by more than 10× on MobileNetV3-Small, whose depthwise convolutions have
   no fast CPU path, yet it is slower on the Mamba model, because a sequential Python loop
   issuing small kernels leaves the GPU idle between launches. Which model benefits depends
   on which trade-offs it happens to need.
8. **Python costs approximately 150 ns per operator.** The C++ timer measures the same
   `x + x` at approximately 245 ns against approximately 397 ns in Python. Multiplied by
   the 27,000 operator calls per Mamba forward pass reported in the profiler notebook, this
   accounts for most of that model's runtime.

---

## References

- [`torch.utils.benchmark` API documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)
- [Valgrind / Callgrind manual](https://valgrind.org/docs/manual/cl-manual.html)
