# Notebook 1 — `pytorch_profiler_workshop.ipynb`

Using `torch.profiler` to determine **where** execution time is spent.

[← back to README](../README.md) · [self-study guide](self-study-guide.md) · [models](models.md) · [benchmark notebook](benchmark-notebook.md) · [trace analysis](trace-analysis.md)

---

## Contents

| Part | Topic | Approx. time |
|---|---|---|
| **0** | Setup, imports and helper functions (`prof_df`, `bench`, `module_df`) | 5 min |
| **1** | Profiler fundamentals: each flag introduced individually | 30 min |
| **2** | **Profiling inference**: architecture, latency, throughput, optimisation | 45 min |
| **3** | **Profiling training**: backward pass, memory, data loading | 35 min |
| **4** | Reading traces: Perfetto, flame graphs, caveats, MPS | 15 min |
| **5** | **Trace analysis**: Chrome Trace parsing, Holistic Trace Analysis, `TraceDiff` | 25 min |
| **6** | Checklist and exercises | 10 min |

Parts 2 and 3 are kept separate by design. Profiling an inference service and profiling a
training loop are distinct tasks with different metrics, different pitfalls and different
remedies. The notebook presents a case in which the same code change improves one and
degrades the other.

---

## Features covered

Each item is demonstrated with runnable code rather than described in passing.

- `profile()` context manager and `ProfilerActivity`
- `record_function()` for custom labels, including nested labels
- **Self time versus total time**, and the consequences of sorting by the wrong one
- `record_shapes=True` with `key_averages(group_by_input_shape=True)`
- `profile_memory=True` for per-operator allocation and release
- `with_stack=True` with `key_averages(group_by_stack_n=k)` for source attribution
- `nn.Module` spans for per-module timing, and the reason `with_modules` is deprecated
- `with_flops=True`, using achieved GFLOP/s to distinguish compute-bound from
  overhead-bound operators
- `export_chrome_trace()` for the Perfetto timeline
- `export_stacks()` for flame graphs
- `schedule(skip_first, wait, warmup, active, repeat)` with `prof.step()`
- `on_trace_ready`, `tensorboard_trace_handler` and `acc_events`
- `export_memory_timeline()`, including its deprecation status in torch 2.14
- `torch.mps.profiler`, and the absence of `ProfilerActivity.MPS`
- **Forward hooks as profiling instruments**: generating `record_function` labels for every
  layer automatically (§2.2.7), which yields a per-layer breakdown of a model one did not
  write
- Trace export and analysis, covered in [trace-analysis.md](trace-analysis.md)

---

## Architecture reference (§2.2)

Each of the four models is documented in three forms before it is profiled, so that the
operator-mix results presented later have a concrete structure to refer back to. Full
architecture notes are in [models.md](models.md).

1. **Mermaid diagrams** showing the shape of the computation, including detail diagrams of
   the ResNet `BasicBlock`, the `ViTBlock`, and the Mamba `S6` mixer, in which the
   sequential scan loop is drawn as an explicit self-edge.
2. **Layer-by-layer tables** generated at runtime with forward hooks (`layer_table()`),
   giving layer name, type, output shape, parameter count and parameter share.
3. **Structure combined with measured cost** (`structure_and_cost()`): the same table with
   `time_ms` and `time_%` columns. This produces the central observation of the section:

   > ResNet-18's `layer4` holds **72% of the parameters but approximately 17% of the
   > time**, while `conv1` and `maxpool` together hold **close to 0% of the parameters and
   > approximately 29% of the time**. Parameters predict memory consumption; activations
   > predict time.

Mermaid diagrams render inline in JupyterLab 4.1+, VS Code and GitHub. Where a viewer
displays the raw code block instead, the diagram can be pasted into <https://mermaid.live>.

---

## Findings expected to reproduce

Absolute timings will differ between machines. The following relationships should not.

1. **Sorting by self time and by total time produces different answers.** Total times sum
   to considerably more than the wall-clock runtime because they double-count nesting.
2. **`with_stack=True` alone records nothing.** It must be accompanied by
   `experimental_config=_ExperimentalConfig(verbose=True)`; otherwise `export_stacks()`
   writes a zero-byte file without reporting an error. Verified on torch 2.14.
3. **MobileNetV3-Small performs roughly 32× less arithmetic than ResNet-18 on the same
   input and takes about 2.5× longer.** Its 52 convolution layers produce roughly 2,400
   convolution kernel calls, because 11 of them are depthwise and decompose into one call
   per channel. It achieves approximately 5 GFLOP/s against ResNet-18's 340.
4. **A `MambaBlock` has approximately 13× fewer parameters than a `ViTBlock` and still
   takes several times longer** (§2.2.7). The architecture table does not account for this;
   the profiler does.
5. **VisionMamba averages approximately 1 µs of work per operator call, against
   approximately 70 µs for ResNet.** This single figure classifies the bottleneck as
   overhead-bound or compute-bound.
6. **Hoisting loop-invariant work out of the scan reduces `aten::exp` from approximately
   260 calls to 8** and makes inference approximately 1.3× faster, with identical
   arithmetic and bit-identical output.
7. **bfloat16 autocast on Apple Silicon CPU is approximately 20× slower.** The profiler
   attributes the additional cost to the convolution kernel rather than to the casts, so
   the common remedy of eliminating casts would not have helped. An optimisation is a
   hypothesis, and the profiler is the experiment that tests it.
8. **A forward pass with gradients enabled retains approximately 165 MB where
   `inference_mode` retains none.** These are the saved activations, and they are the
   principal reason batch size is limited by memory.
9. **The `S6Fast` rewrite from Part 2 makes training slower** (approximately 55 ms against
   48 ms) while allocating 4.5× more memory, because the precomputed tensors become saved
   activations under autograd. Inference improvements do not transfer automatically to
   training.
10. **The first MPS call takes approximately 10× the steady-state time**, owing to Metal
   pipeline compilation, against approximately 1.1× for CPU eager execution. The importance
   of warmup varies considerably by backend.

---

## References

- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [`torch.profiler` API documentation](https://docs.pytorch.org/docs/stable/profiler.html)
- [FlameGraph](https://github.com/brendangregg/FlameGraph), which renders `export_stacks()` output
