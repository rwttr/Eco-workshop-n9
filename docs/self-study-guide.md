# Self-study guide

Working through both notebooks without an instructor.

[← README](../README.md) · [models](models.md) · [profiler notebook](profiler-notebook.md) · [benchmark notebook](benchmark-notebook.md) · [trace analysis](trace-analysis.md)

## How to use this guide

Run the notebooks in order. Each session below adds what an instructor would:

- **Goal**: what you can do afterwards.
- **Work through**: which sections to run.
- **Look for**: the figures that carry the lesson, often not the biggest number on screen.
- **Checkpoint**: questions to answer *before* expanding the answer. Predicting first, and
  being wrong, is what makes these lessons stick.
- **Common confusion**: the usual mistake at this point.

Plan on six sessions of 45–60 minutes. The notebooks run in ~3 minutes total; the rest is
reading output and thinking.

> Answers are in collapsible blocks (they expand in GitHub, VS Code and most Markdown
> viewers). If yours shows them open, cover the screen below the question.

## Before you start

Run Part 0 of `pytorch_profiler_workshop.ipynb` and check:

- `torch` is 2.6 or later.
- `CPU threads` is above 1. Note it: every timing depends on it.
- `MPS available` is `True` on Apple Silicon, `False` elsewhere. Either is fine; MPS cells
  skip themselves.

You should already know basic PyTorch: an `nn.Module`, a training loop with `loss.backward()`
and `optimizer.step()`, and `model.eval()` vs `model.train()`. The workshops focus on
measurement.

## Study plan

| Session | Material | Central idea |
|---|---|---|
| 1 | Notebook 1, Parts 0–1 | Self time and total time answer different questions |
| 2 | Notebook 1, §2.1–2.5 | Neither parameters nor FLOPs predict cost; measured operator mix does |
| 3 | Notebook 1, §2.6–2.9 | An optimisation is a hypothesis to test |
| 4 | Notebook 1, Part 3 | Training has different bottlenecks from inference |
| 5 | Notebook 1, Parts 4–5 | Traces keep the time order that aggregates throw away |
| 6 | Notebook 2, all parts | A measurement without an error bar is not a result |

---

## Session 1 — Profiler fundamentals

**Goal.** Read a profiler table correctly, and attribute time to your own code rather than
to operator names.

**Work through.** Notebook 1, Parts 0 and 1 (§1.1–§1.10).

**Look for.** In §1.2 the same run is shown sorted two ways. Compare the two sums at the
bottom: self times add up to about the real runtime; total times add up to several times
that, because a parent's total includes its children. In §1.3, `record_function` labels
make the report use your names (`stem`, `layer1`, `head`) instead of ATen ops, which is what
makes a profile readable in a large codebase.

**Checkpoint.**

<details>
<summary>1. <code>aten::conv2d</code> shows a large total time and near-zero self time. What is it doing?</summary>

Nothing, computationally. It is a dispatcher wrapper: it calls `aten::convolution` →
`aten::_convolution` → the real kernel (`aten::_slow_conv2d_forward` or
`aten::mkldnn_convolution`, depending on the build). Its total is large because it contains
its children; its self time is near zero because it does no work. Sorting by total and
optimising the top row would mean optimising a function that does nothing.
</details>

<details>
<summary>2. You want the cost of a transformer's attention block. Sort by self time or total time?</summary>

Total time, on a `record_function` label wrapping the block. A label owns everything beneath
it, so total time is the right measure for a subsystem. Self time answers a different
question, which single kernel is hot; use it when hunting a hotspot, not costing a component.
</details>

<details>
<summary>3. You set <code>with_stack=True</code>, call <code>export_stacks()</code>, and get a zero-byte file. Why?</summary>

In torch 2.14, `with_stack=True` also needs
`experimental_config=_ExperimentalConfig(verbose=True)` before the stack field is populated.
Without it, `key_averages(group_by_stack_n=...)` returns empty stacks and `export_stacks()`
silently writes an empty file. §1.6 shows the failure before the fix, because the failure
gives no hint of its cause.
</details>

**Common confusion.** Percentages in a total-time table can exceed 100%. That is correct,
not a bug: nesting is counted more than once. Only self times sum to the runtime.

---

## Session 2 — Architecture vs measured cost

**Goal.** Explain why parameter count predicts latency badly, and tell from a profile
whether a model is limited by arithmetic or by operator dispatch.

**Work through.** Notebook 1, §2.1–§2.5. Don't skip the diagrams in §2.2.1–§2.2.3; the
operator-mix results are hard to read without knowing each model's shape.

**Look for.** §2.2.6 prints `params_%` and `time_%` side by side for ResNet-18: read them
against each other. §2.4 computes mean µs per operator call, the diagnostic all of Part 2
rests on.

**Checkpoint.**

<details>
<summary>1. Before running §2.2.6: which ResNet-18 layer holds the most parameters, and which takes the most time?</summary>

`layer4` holds ~72% of the parameters, and most people answer `layer4` to both. It takes
only ~17% of the time. `conv1` + `maxpool` hold ~0% of the parameters but ~29% of the time,
because they run at 112×112 and 56×56 while `layer4` runs on 7×7.

Parameters predict memory. Activation sizes predict time.
</details>

<details>
<summary>2. Rank the three models by latency, before running §2.2.7.</summary>

Parameters (11.7 M, 2.9 M, 2.5 M) and FLOPs (14.5, 4.4, 0.45 G) agree: ResNet-18, then
ViT-Tiny, then MobileNetV3-Small. Latency does not. Measured on the reference machine:
ViT-Tiny ~10 ms, ResNet-18 ~36 ms, **MobileNetV3-Small ~91 ms**.

MobileNetV3-Small does ~32× less arithmetic than ResNet-18 on the same input and takes
~2.5× longer: it goes from cheapest on paper to slowest on the clock.
</details>

<details>
<summary>3. MobileNetV3-Small was designed for efficiency and does far less arithmetic than ResNet-18. Why is it the slowest here?</summary>

Because of how its depthwise convolutions *execute*, not how much arithmetic they need. With
`groups = channels`, each channel has its own filter, and without a fused depthwise kernel
PyTorch loops over the groups.

The profiler shows it: 52 `Conv2d` layers produce ~2,400 conv kernel calls, ~215 per
depthwise layer, up to 576 groups in one layer. Each call is a dispatcher round-trip plus an
allocation for a one-channel tensor. Achieved GFLOP/s says the same: ~5 vs ResNet-18's 340.
It does 32× less arithmetic ~68× less efficiently.

This is not a criticism of the architecture. On MPS it runs in ~8 ms vs ~90 ms on CPU, the
biggest device speedup in the set. It was built for mobile accelerators; a laptop CPU is the
wrong hardware.
</details>

<details>
<summary>4. MobileNetV3-Small averages ~2 µs of work per operator call; ResNet-18 and ViT-Tiny average 15–70 µs. What does that tell you, and what follows?</summary>

PyTorch's dispatch and allocation overhead is ~1–3 µs per operator. At ~2 µs of work per
call, a model spends most of its time in the framework: it is **overhead-bound**. Faster
arithmetic won't help; issue fewer, larger ops. At tens of µs per call a model is
**compute-bound**, and the arithmetic, kernel or dtype is worth attention.

The column classifies the model, but it does not name the cause. For MobileNetV3-Small that
takes the operator table: the cause is a kernel decomposition below the architecture, not
anything visible in the model code.
</details>

**Common confusion.** Module count is not operator count. MobileNetV3-Small has the most
leaf modules, but its ~41,000 operator calls come from 11 depthwise layers splitting into one
call per channel, which no module census shows.

---

## Session 3 — Testing an optimisation

**Goal.** Use a profile to propose a change, then confirm or reject it by measurement.

**Work through.** Notebook 1, §2.6–§2.9.

**Look for.** §2.6's profile diff prints operators that got cheaper *and* ones that got
dearer. Read both: the second is where the optimisation's cost shows. §2.8 C is the most
important cell in Part 2; read the operator table, not just the wall-clock numbers.

**Checkpoint.**

<details>
<summary>1. Fused attention cuts ViT-Tiny's operator calls by ~70%, yet inference is only ~1.2× faster. Why don't the two ratios match?</summary>

Most of the ops that disappeared were cheap: indexing, views and reshapes that cost well
under a microsecond each. The diff shows where the time actually went: `aten::bmm` and
`aten::_softmax`, the two score-matrix multiplies and the softmax between them, are gone,
along with most of the `aten::copy_` calls around them. In their place is one
`aten::_scaled_dot_product_flash_attention_for_cpu` per block, which still does the
arithmetic.

The principle is not "fewer operations" but "a chain of small ops and their intermediate
tensors replaced by one larger op".
</details>

<details>
<summary>2. bfloat16 autocast makes ResNet-18 ~20× slower here. Propose a fix from that number alone, then check it against the profiler table.</summary>

The usual guess is to avoid repeated casts, e.g. convert weights once. The table shows that
won't help: the extra time is inside `aten::_slow_conv2d_forward` (~22 ms → ~700 ms), not in
`aten::to` or `aten::_to_copy`. There is no fast bfloat16 conv kernel for this hardware, so it
falls back to a slow reference path. Abandon the optimisation on this platform; don't refine it.

An optimisation is a hypothesis; the profiler is the experiment that separates competing
explanations of the same wall-clock number.
</details>

**Common confusion.** A profiler total is not a latency. It includes tracing overhead, which
is large for operator-heavy models. Confirm every improvement with a separate wall-clock
timer; that is what `bench()` is for.

---

## Session 4 — Training

**Goal.** Account for a training step's time and memory, and recognise an input-bound loop.

**Work through.** Notebook 1, Part 3.

**Look for.** §3.3 prints allocated, freed and net retained memory for three configurations.
`net_retained_MB` is what matters; the allocated column is nearly the same in all three and
misleading alone. §3.4 labels data-loading wait and compute separately; compare their shares
before and after adding workers.

**Checkpoint.**

<details>
<summary>1. A forward pass under <code>inference_mode</code> and one with gradients enabled allocate almost the same memory. Why is only one a problem?</summary>

It is about what gets *freed*. Under `inference_mode`, allocated equals freed and net
retention is ~0 MB: each intermediate is released once consumed. With gradients on, ~165 MB
is never freed: the saved activations, held until `loss.backward()`. That is why memory
limits batch size in training but not inference, and what gradient checkpointing and
gradient accumulation address.
</details>

<details>
<summary>2. After adding <code>num_workers=4</code>, <code>DATA_WAIT</code> still has the larger share. Did the fix fail?</summary>

No. The absolute wait drops ~4×, from ~263 ms to ~59 ms. The share stays high because the
demo model is deliberately trivial, leaving little compute to overlap with loading.

Track `COMPUTE / (COMPUTE + DATA_WAIT)`, your pipeline efficiency, which rises sharply. If it
is still low after adding workers, the dataset is too slow and `__getitem__` needs fixing.
</details>

<details>
<summary>3. Fused attention was ~1.2× faster for inference. Why is its training speed-up smaller, and what does it still win?</summary>

Compare `fwd_ms` and `bwd_ms` in §3.5: fusion speeds up the forward pass, but the backward
pass costs about the same either way, so a full step gains less (~23 vs ~25 ms). What it
still wins is memory: 183 vs 260 MB allocated. The hand-written version saves every
`197 × 197` score matrix and softmax for backward; the fused kernel does not.

An optimisation's payoff depends on the mode. Profile the one you ship.
</details>

**Common confusion.** "Backward costs 2× forward" is a rule of thumb. Measured ratios here
range ~1.3–1.9. Read your own number; a ratio well above 2 means the backward graph has
something worth investigating.

---

## Session 5 — Traces

**Goal.** Capture a trace other tools can read, inspect it visually, and diff two traces to
find what changed.

**Work through.** Notebook 1, Parts 4 and 5, with [trace-analysis.md](trace-analysis.md)
open for the screenshots and capture gotchas.

**Look for.** §5.2 checks a hand-written trace parser against `prof.key_averages()`; they
should agree within run-to-run variation, proving the trace holds everything the aggregate
does. §5.5 diffs a baseline against a deliberately regressed trace; in real investigations
the `added` row of `ops_diff` is often the whole answer.

**Checkpoint.**

<details>
<summary>1. You save a trace with <code>prof.export_chrome_trace(...)</code> from a scheduled profiler, and HTA won't parse it. Why?</summary>

It has no `ProfilerStep#N` markers. Those are written only via `tensorboard_trace_handler`,
not a manual `export_chrome_trace()`, even with a `schedule`. HTA reports it as
`AttributeError: Can only use .str accessor with string values`, which describes an empty
symbol lookup, not the real cause. `ws_trace.capture_trace()` captures correctly.
</details>

<details>
<summary>2. In a trace diff, <code>ConvolutionBackward0</code> is 3% slower with the same call count, after a change that didn't touch convolutions. Is this a finding?</summary>

No, it's noise. Trace-diff durations come from one profiled run and include profiler
overhead, and HTA's self-time approximation on deep CPU traces can even go negative. Call
counts are exact: capture twice and the integers match. Decide *whether* something changed
with a benchmark and error bars; decide *what* changed with `diff_counts`.
</details>

**Common confusion.** HTA is built for multi-GPU distributed jobs. On a CPU-only trace most
headline analyses have no data and return empty or raise. The parsing and diffing layer
works, and on CPU that is the reason to install it.

---

## Session 6 — Benchmarking

**Goal.** Produce a timing you would defend, and decide whether a difference is real.

**Work through.** Notebook 2, all parts. §1.4 and §3.3 are the core; the rest applies them.

**Look for.** §1.4 benchmarks identical work twice: the gap is your machine's noise floor,
and every later claim must beat it. §5.3 measures one comparison two ways, variants run in
turn vs interleaved. They disagree, and interleaved is right.

**Checkpoint.**

<details>
<summary>1. You time an MPS matmul with <code>time.perf_counter()</code> and get 0.03 ms for work that takes 6 ms. What happened?</summary>

Accelerator work is asynchronous: the Python call returns once the work is queued, so you
timed filling a queue. `torch.utils.benchmark.Timer` avoids this because its default clock
calls `torch.accelerator.synchronize()` first. That alone is a strong reason to use `Timer`
over a hand-written loop.
</details>

<details>
<summary>2. You measure A three times, then B three times, and B looks 4% faster. What's wrong with this?</summary>

Any drift during the cell (thermals, another process, frequency scaling) is blamed entirely
on B, because B ran later, and 4% is well within tens of seconds of drift. Interleaving
(A, B, A, B, A, B) spreads drift across both. The notebook keeps the flawed version next to
the fix, because it produced a confidently wrong ordering.
</details>

<details>
<summary>3. <code>set_to_none=True</code> is ~5% faster. Is that why it's the default?</summary>

No. The time gap is small and would vanish on a noisier machine. The real difference is
memory: `set_to_none=False` keeps a gradient tensor per parameter, ~43 MB here, for the
whole run; `set_to_none=True` frees them. It is a memory optimisation usually described as a
speed one, and measuring is how you find out.
</details>

**Common confusion.** `timeit(n)` returns one replicate, so no error bar.
`blocked_autorange(min_run_time=t)` picks the iteration count and returns many replicates,
which gives you an IQR. Prefer it unless you have a reason not to.

---

## Self-assessment

You have it if you can do these without looking back:

- [ ] Say what self time and total time measure, and which to sort by for a given question.
- [ ] Name the flags needed for `export_stacks()` to produce output.
- [ ] Explain why parameter count doesn't predict latency, using ResNet-18.
- [ ] Explain why FLOP count doesn't predict latency, using MobileNetV3-Small.
- [ ] Classify a model as overhead- or compute-bound from one profile column.
- [ ] Explain why a forward pass retains memory with gradients enabled.
- [ ] Describe an optimisation that helps inference and hurts training, and why.
- [ ] Capture a trace HTA can read, and say what makes it readable.
- [ ] State your machine's benchmark noise floor and how you measured it.
- [ ] Explain why to interleave measurements rather than group them.
- [ ] Say which trace-diff column to trust, and why not the other.

## Applying this to your own work

Both notebooks end with the same exercise, and it is the point of the workshops:

1. Take a model from your own project.
2. Benchmark it with `Timer.blocked_autorange()`; record median and IQR.
3. Profile it; classify the bottleneck by mean µs per operator.
4. Change exactly one thing.
5. Re-benchmark; use `ab_compare()` to check the change beats your noise floor.
6. If it does, trace before and after and run `TraceDiff.ops_diff` to see which operators
   account for it.

Step 6 is the one people skip, and it is what turns one result into something you can reuse
on the next model. A number shows a change worked; a diff shows *what kind* of change works.

Then write three sentences: what was slow, what kind of slow it was, and what you changed.
If you can't write the second, go back to step 3.
