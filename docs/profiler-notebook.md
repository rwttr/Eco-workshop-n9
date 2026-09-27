# Notebook 1 — `pytorch_profiler_workshop.ipynb`

`torch.profiler`: **where** execution time goes.

[← README](../README.md) · [self-study guide](self-study-guide.md) · [models](models.md) · [benchmark notebook](benchmark-notebook.md) · [trace analysis](trace-analysis.md)

## Contents

| Part | Topic | Time |
|---|---|---|
| **0** | Setup and helpers (`prof_df`, `bench`, `module_df`) | 5 min |
| **1** | Profiler fundamentals, one flag at a time | 30 min |
| **2** | **Inference**: architecture, latency, throughput, optimisation | 45 min |
| **3** | **Training**: backward pass, memory, data loading | 35 min |
| **4** | Reading traces: Perfetto, flame graphs, caveats, MPS | 15 min |
| **5** | **Trace analysis**: Chrome Trace parsing, HTA, `TraceDiff` | 25 min |
| **6** | Checklist and exercises | 10 min |

Inference and training are separate parts on purpose: they have different metrics, pitfalls
and fixes, and the notebook shows one code change that helps the first and hurts the second.

## Features covered

Each is demonstrated with runnable code.

- `profile()` and `ProfilerActivity`
- `record_function()` labels, including nested ones
- **Self time vs total time**, and what goes wrong when you sort by the wrong one
- `record_shapes=True` with `key_averages(group_by_input_shape=True)`
- `profile_memory=True`: per-operator allocation and release
- `with_stack=True` with `key_averages(group_by_stack_n=k)` for source attribution
- `nn.Module` spans for per-module timing, and why `with_modules` is deprecated
- `with_flops=True`: achieved GFLOP/s separates compute-bound from overhead-bound operators
- `export_chrome_trace()` (Perfetto) and `export_stacks()` (flame graphs)
- `schedule(skip_first, wait, warmup, active, repeat)` with `prof.step()`
- `on_trace_ready`, `tensorboard_trace_handler`, `acc_events`
- `export_memory_timeline()` and its deprecation in torch 2.14
- `torch.mps.profiler`, and the absence of `ProfilerActivity.MPS`
- **Forward hooks as instruments**: auto-generated `record_function` labels for every layer
  (§2.2.7) give a per-layer breakdown of a model you did not write
- Trace export and analysis ([trace-analysis.md](trace-analysis.md))

## Architecture reference (§2.2)

Before profiling, each model is shown three ways (full notes in [models.md](models.md)):

1. **Mermaid diagrams**, including detail views of the ResNet `BasicBlock`, the `ViTBlock` and
   the Mamba `S6` mixer, whose sequential scan is drawn as a self-edge.
2. **Layer tables** from forward hooks (`layer_table()`): name, type, output shape, parameters.
3. **Structure plus measured cost** (`structure_and_cost()`): the same table with `time_ms` and
   `time_%`. Key result: ResNet-18's `layer4` holds **72% of the parameters but ~17% of the
   time**; `conv1` + `maxpool` hold **~0% of the parameters but ~29% of the time**.

Mermaid renders in JupyterLab 4.1+, VS Code and GitHub; otherwise paste into <https://mermaid.live>.

## Findings expected to reproduce

Absolute timings vary by machine; these relationships should not.

1. **Self-time and total-time sorts disagree.** Total times sum to far more than wall-clock
   because nesting is double-counted.
2. **`with_stack=True` alone records nothing.** It also needs
   `experimental_config=_ExperimentalConfig(verbose=True)`, or `export_stacks()` silently
   writes a zero-byte file (torch 2.14).
3. **MobileNetV3-Small: ~32× less arithmetic than ResNet-18, ~2.5× slower.** Its 52 conv layers
   make ~2,400 conv kernel calls because the 11 depthwise ones run one call per channel. It
   achieves ~5 GFLOP/s vs ResNet-18's 340.
4. **A `MambaBlock` has ~13× fewer parameters than a `ViTBlock` yet takes several times
   longer** (§2.2.7). The architecture table can't explain this; the profiler can.
5. **VisionMamba averages ~1 µs of work per operator call; ResNet ~70 µs.** That one figure
   classifies a model as overhead-bound or compute-bound.
6. **Hoisting loop-invariant work out of the scan cuts `aten::exp` from ~260 calls to 8** and
   speeds inference ~1.3×, with identical arithmetic and bit-identical output.
7. **bfloat16 autocast on Apple Silicon CPU is ~20× slower.** The extra cost is in the conv
   kernel, not the casts, so removing casts would not help. An optimisation is a hypothesis;
   the profiler tests it.
8. **With gradients enabled, a forward pass retains ~165 MB; `inference_mode` retains none.**
   Those are saved activations, the main reason memory limits batch size.
9. **`S6Fast` makes training slower** (~55 ms vs 48 ms) with 4.5× the memory, because its
   precomputed tensors become saved activations. Inference wins don't transfer automatically.
10. **The first MPS call takes ~10× steady-state time** (Metal pipeline compilation) vs ~1.1×
    for CPU eager. How much warmup matters depends on the backend.

## References

- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) · [`torch.profiler` API](https://docs.pytorch.org/docs/stable/profiler.html)
- [FlameGraph](https://github.com/brendangregg/FlameGraph), for rendering `export_stacks()` output
