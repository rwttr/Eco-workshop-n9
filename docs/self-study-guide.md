# Self-study guide

A follow-through manual for working through both notebooks without an instructor.

[← back to README](../README.md) · [models](models.md) · [profiler notebook](profiler-notebook.md) · [benchmark notebook](benchmark-notebook.md) · [trace analysis](trace-analysis.md)

---

## How to use this guide

The notebooks are written to be read and executed in order. This guide adds what an
instructor would normally supply: what each section is for, what to look at in the output,
and a checkpoint to confirm that the point landed before moving on.

Each session below follows the same shape.

- **Goal** — what you should be able to do afterwards.
- **Work through** — which notebook sections to run.
- **What to look for** — the specific columns or figures that carry the lesson, which is
  usually not the largest number on screen.
- **Checkpoint** — questions to answer before continuing. Attempt them before expanding
  the answer; predicting first and being wrong is the mechanism by which these particular
  lessons stick.
- **Common confusion** — the mistake most people make at this point.

Expect roughly six sessions of 45–60 minutes. The notebooks themselves execute in about
three minutes in total; the remaining time is reading output and thinking about it.

> Answers are written inside collapsible blocks. These expand in GitHub, VS Code and most
> Markdown viewers. If yours displays them already expanded, cover the screen below the
> question before reading on.

---

## Before you start

Confirm the environment is working. In `pytorch_profiler_workshop.ipynb`, run Part 0 and
check the printed summary:

- `torch` reports 2.6 or later.
- `CPU threads` shows a number greater than 1. Record it, because every timing in both
  notebooks depends on it.
- `MPS available` is `True` on Apple Silicon and `False` elsewhere. Either is fine; the
  MPS cells skip themselves cleanly.

You also need some familiarity with PyTorch itself: defining an `nn.Module`, a training
loop with `loss.backward()` and `optimizer.step()`, and the difference between `model.eval()`
and `model.train()`. The workshops assume this and concentrate on measurement.

---

## Study plan

| Session | Material | Central idea |
|---|---|---|
| 1 | Notebook 1, Parts 0–1 | The profiler reports self time and total time, and they answer different questions |
| 2 | Notebook 1, Part 2.1–2.5 | Neither parameters nor FLOPs predict cost; measured operator mix does |
| 3 | Notebook 1, Part 2.6–2.9 | An optimisation is a hypothesis that must be tested |
| 4 | Notebook 1, Part 3 | Training has different bottlenecks from inference |
| 5 | Notebook 1, Parts 4–5 | Traces preserve time order, which aggregates discard |
| 6 | Notebook 2, all parts | A measurement without an error bar is not a result |

---

## Session 1 — Profiler fundamentals

**Goal.** Read a profiler table correctly and attribute time to your own code rather than
to operator names.

**Work through.** Notebook 1, Part 0 and Part 1 (§1.1–§1.10).

**What to look for.**

In §1.2 the same profiling run is displayed twice, sorted two different ways. Compare the
two printed sums at the bottom of the cell. The self times add up to approximately the real
runtime; the total times add up to several times that figure, because a parent operator's
total includes all of its children.

In §1.3, after `record_function` labels are added, the report uses names from the notebook
(`stem`, `layer1`, `head`) rather than ATen operator names. This is the mechanism by which
a profiler report becomes readable in a codebase of any size.

**Checkpoint.**

<details>
<summary>1. <code>aten::conv2d</code> shows a large total time and a self time close to zero. What is it doing?</summary>

Nothing, computationally. It is a dispatcher wrapper that calls `aten::convolution`, which
calls `aten::_convolution`, which calls the kernel that performs the arithmetic
(`aten::_slow_conv2d_forward` or `aten::mkldnn_convolution`, depending on the build). Its
total time is large because it contains its children; its self time is near zero because it
performs no work itself.

Sorting by total time and then optimising the top row would mean optimising a function that
does nothing.
</details>

<details>
<summary>2. You want to know how much the attention block of a transformer costs. Do you sort by self time or total time?</summary>

Total time, applied to a `record_function` label wrapping the block. A label's purpose is to
own everything beneath it, so total time is the correct measure for a subsystem.

Self time answers a different question: which individual kernel is consuming CPU. Use that
one when you are looking for the hotspot rather than costing a component.
</details>

<details>
<summary>3. You set <code>with_stack=True</code>, call <code>export_stacks()</code>, and get a zero-byte file. Why?</summary>

`with_stack=True` is not sufficient on its own in torch 2.14. The profiler also requires
`experimental_config=_ExperimentalConfig(verbose=True)` before it will populate the stack
field. Without it, `key_averages(group_by_stack_n=...)` returns rows with empty stacks and
`export_stacks()` writes an empty file, reporting no error.

Section 1.6 demonstrates the failure before the correction, because the failure mode gives
no indication of its own cause.
</details>

**Common confusion.** The percentages in a table sorted by total time can exceed 100%. This
is correct behaviour, not a bug: nesting is counted more than once. Only self times sum to
the runtime.

---

## Session 2 — Architecture against measured cost

**Goal.** Explain why parameter count is a poor predictor of latency, and identify from a
profile whether a model is limited by arithmetic or by operator dispatch.

**Work through.** Notebook 1, §2.1 through §2.5. Do not skip the diagrams in §2.2.1–§2.2.4;
the operator-mix results later are difficult to interpret without knowing the shape of each
model.

**What to look for.**

Section §2.2.7 prints a table with `params_%` and `time_%` side by side for ResNet-18.
Read those two columns against each other rather than reading either alone.

Section §2.4 computes mean microseconds per operator call. This single figure is the
diagnostic the whole of Part 2 depends on.

**Checkpoint.**

<details>
<summary>1. Before running §2.2.7: which ResNet-18 layer holds the most parameters, and which accounts for the most time?</summary>

`layer4` holds approximately 72% of the parameters, and most people answer `layer4` to both
questions. It accounts for only about 17% of the time.

The time is distributed very differently. `conv1` and `maxpool` together hold close to 0%
of the parameters and account for roughly 29% of the time, because they operate at
112×112 and 56×56 on a full-resolution image, whereas `layer4` operates on a 7×7 feature
map.

Parameters predict memory consumption. Activation sizes predict time.
</details>

<details>
<summary>2. Rank all four models by latency, before running §2.2.8.</summary>

Ranking by parameter count (11.7 M, 2.5 M, 2.9 M, 0.16 M) gives the wrong order, and so
does ranking by FLOPs. The measured order on the reference machine is ViT-Tiny fastest at
approximately 10 ms, then VisionMamba-Tiny at approximately 21 ms, then ResNet-18 at
approximately 36 ms, and **MobileNetV3-Small slowest at approximately 91 ms**.

Two results are worth dwelling on. MobileNetV3-Small does roughly 32× less arithmetic than
ResNet-18 on the same input and takes about 2.5× longer. VisionMamba-Tiny has roughly 70×
fewer parameters than ResNet-18 and processes 12× fewer pixels, yet takes about twice as
long as the ViT.
</details>

<details>
<summary>3. MobileNetV3-Small was designed for efficiency and performs far less arithmetic than ResNet-18. Why is it the slowest model here?</summary>

Because of how its depthwise convolutions are executed rather than how much arithmetic they
require. A depthwise convolution sets `groups = channels`, so each channel gets its own
single-channel filter. Where the build provides no fused depthwise kernel, PyTorch
implements grouped convolution by iterating over the groups.

The profiler makes this visible: 52 `Conv2d` layers produce roughly 2,400 convolution
kernel calls, about 215 per depthwise layer, with one layer reaching 576 groups. Each call
is a dispatcher round-trip plus an allocation for a tensor containing a single channel.

The achieved-GFLOP/s column states the same thing differently: approximately 5 GFLOP/s
against ResNet-18's 340, on the same CPU, in the same process. It does 32× less arithmetic
roughly 68× less efficiently.

Note what this is *not*: a criticism of the architecture. On MPS the same model runs in
about 8 ms against 90 ms on the CPU, the largest device speedup in the set. It was designed
for mobile accelerators, and this laptop CPU is the wrong hardware for it.
</details>

<details>
<summary>4. VisionMamba and MobileNetV3-Small average 1–2 µs of work per operator call; ResNet and ViT average 15–90 µs. What does this tell you, and what follows from it?</summary>

PyTorch's dispatch and allocation overhead is on the order of 1–3 µs per operator. A model
averaging 1–2 µs of actual work per call is therefore spending most of its time in the
framework rather than in arithmetic: it is overhead-bound.

The practical consequence is that making the arithmetic faster will not help. The remedy is
to issue fewer, larger operations. A model averaging tens of microseconds per call is
compute-bound, and there the arithmetic, the kernel or the data type is worth attention.

Note that the two overhead-bound models arrive there by different routes: VisionMamba
through an explicit Python loop in code you can see and edit, MobileNetV3-Small through a
kernel decomposition happening below the level of the architecture. The single column
classifies both; identifying the cause still requires reading the operator table.
</details>

**Common confusion.** Module count is not operator count, in either direction.
VisionMamba-Tiny has the fewest leaf modules of the four and issues about 27,000 operator
calls, because a single `S6Naive` module contains a Python loop over 64 timesteps.
MobileNetV3-Small has the most leaf modules and issues about 41,000 calls, for an unrelated
reason. Neither figure is visible in a module census.

---

## Session 3 — Testing an optimisation

**Goal.** Use a profile to propose a change, then confirm or reject it by measurement.

**Work through.** Notebook 1, §2.6 through §2.9.

**What to look for.**

In §2.6 the profile diff prints two tables: operators that became cheaper, and operators
that became more expensive. Read both. The second table is where the cost of the
optimisation appears.

Section §2.8 C is the most important cell in Part 2. Read the operator table, not only the
wall-clock figures.

**Checkpoint.**

<details>
<summary>1. The <code>S6Fast</code> rewrite reduces total operator calls by only about 10%, yet is roughly 1.35× faster. How?</summary>

Total operator count is the wrong measure here. The diff shows `aten::exp` falling from
approximately 260 calls to 8, and `aten::matmul` from 264 to 8, because the rewrite computes
`dA` and `dBu` for every timestep at once rather than once per loop iteration.

The same number of exponentials is computed. What disappeared is roughly 250 dispatcher
round-trips and 250 output allocations. The remaining call count is dominated by cheap
indexing operations such as `aten::select`, which are nearly free.

So the principle is not "fewer operations" but "the expensive operations replaced by fewer,
larger ones".
</details>

<details>
<summary>2. bfloat16 autocast makes ResNet-18 about 20× slower on this machine. Propose a fix from that figure alone, then check it against the profiler table.</summary>

The common proposal is to avoid the repeated casts, for example by converting the weights
once rather than on every call. The profiler table shows that this would not help.

The additional time appears inside `aten::_slow_conv2d_forward` itself, which rises from
approximately 22 ms to approximately 110 ms. It is not in `aten::to` or `aten::_to_copy`.
There is no fast bfloat16 convolution kernel for this hardware, so the operation falls back
to a slow reference implementation. The optimisation should be abandoned on this platform
rather than refined.

This is the general point: an optimisation is a hypothesis, and the profiler is the
experiment that distinguishes between competing explanations of the same wall-clock number.
</details>

**Common confusion.** A profiler total is not a latency figure. It includes tracing
overhead, which is substantial for an operator-heavy model. Confirm every improvement with
a separate wall-clock timer, which is what `bench()` is for.

---

## Session 4 — Training

**Goal.** Account for where a training step spends its time and its memory, and recognise
an input-bound training loop.

**Work through.** Notebook 1, Part 3.

**What to look for.**

Section §3.3 prints allocated, freed and net retained memory for three configurations. The
`net_retained_MB` column is the one that matters; the allocated column is nearly identical
in all three cases and is misleading on its own.

Section §3.4 labels the data-loading wait and the compute separately. Compare their shares
before and after worker processes are introduced.

**Checkpoint.**

<details>
<summary>1. A forward pass under <code>inference_mode</code> and the same forward pass with gradients enabled allocate almost the same amount of memory. Why is only one of them a problem?</summary>

Because of what is freed rather than what is allocated. Under `inference_mode`, allocated
and freed are equal and net retention is approximately 0 MB: each intermediate is released
as soon as it is consumed.

With gradients enabled, approximately 165 MB is never freed. Those are the saved
activations, held until `loss.backward()` consumes them. This is why batch size is limited
by memory in training and not in inference, and it is the problem that gradient
checkpointing and gradient accumulation address.
</details>

<details>
<summary>2. After adding <code>num_workers=4</code>, <code>DATA_WAIT</code> still holds the larger share of the time. Did the fix fail?</summary>

No. The absolute wait falls by roughly a factor of four, from about 263 ms to about 59 ms.
The share remains high because the model in that demonstration is deliberately trivial, so
there is very little compute to overlap the loading with.

The figure to track is the ratio `COMPUTE / (COMPUTE + DATA_WAIT)`, which is your pipeline
efficiency, and it rises sharply. If that ratio is still low after adding workers, the
dataset itself is too slow and `__getitem__` is the thing to fix.
</details>

<details>
<summary>3. <code>S6Fast</code> was clearly faster for inference. Why is it slower in training?</summary>

The rewrite precomputes `dA` and `dBu` as full `(B, L, D, N)` tensors. Under
`inference_mode` these are transient: built, consumed, freed.

Under autograd they become saved activations. They must survive until the backward pass,
they must be differentiated through, and they are far larger than the per-step slices the
naive version created. The result is approximately 55 ms per step against 48 ms, with
roughly 4.5× the memory.

An inference optimisation is not automatically a training optimisation. Anything that
materialises large intermediates in order to save dispatcher calls is betting that those
intermediates are short-lived, and autograd voids that bet.
</details>

**Common confusion.** Backward is often described as costing twice the forward pass. The
measured ratios across the models here range from roughly 1.3 to 2.1, and the `S6Fast`
variant reaches 3.0. Read your own figure rather than the rule of thumb; a ratio far above
2 indicates that the backward graph contains something worth investigating, which is
exactly what happens to `S6Fast`.

---

## Session 5 — Traces

**Goal.** Capture a trace that downstream tools can read, inspect it visually, and diff two
traces to identify what changed.

**Work through.** Notebook 1, Part 4 and Part 5. Read
[trace-analysis.md](trace-analysis.md) alongside them; it contains the screenshots and the
two capture requirements that are easy to overlook.

**What to look for.**

Part 5.2 cross-checks a hand-written trace parser against `prof.key_averages()`. The two
should agree to within run-to-run variation. This establishes that the trace file contains
everything the aggregate table does.

Section §5.5 diffs a baseline trace against a deliberately regressed one. The `added` row of
`ops_diff` is frequently the entire answer in a real investigation.

**Checkpoint.**

<details>
<summary>1. You capture a trace with <code>prof.export_chrome_trace(...)</code> from a scheduled profiler, and HTA refuses to parse it. Why?</summary>

The file contains no `ProfilerStep#N` markers. Those are written only when the file is
produced by `tensorboard_trace_handler`, not by a manual `export_chrome_trace()` call, even
when a `schedule` is in use.

HTA reports this as `AttributeError: Can only use .str accessor with string values`, which
describes an empty symbol lookup rather than the actual cause. `ws_trace.capture_trace()`
performs the capture correctly.
</details>

<details>
<summary>2. In a trace diff, <code>ConvolutionBackward0</code> shows a 3% duration increase with an unchanged call count, in a change that did not touch convolutions. Is this a finding?</summary>

No. It is noise. Durations in a trace diff come from a single profiled run and carry the
profiler's overhead, and HTA's self-time approximation over a deeply nested CPU trace can
even produce negative values.

Call counts are exact and deterministic: run the capture twice and the integers are
identical. Decide *whether* something changed using a benchmark with error bars, and *what*
changed using `diff_counts`.
</details>

**Common confusion.** HTA is designed for multi-GPU distributed jobs. On a CPU-only trace,
most of its headline analyses have no data to work with and will return empty results or
raise. The parsing and diffing layer works, and on CPU that is the reason to install it.

---

## Session 6 — Benchmarking

**Goal.** Produce a timing result you are willing to defend, and determine whether a
difference between two results is real.

**Work through.** Notebook 2, all parts. Sections 1.4 and 3.3 are the core; the rest applies
them.

**What to look for.**

Section §1.4 benchmarks identical work twice. The difference between the two results is your
machine's noise floor, and every later claim must exceed it.

Section §5.3 measures the same comparison in two ways: each variant run to completion in
turn, and the variants interleaved. The two methods disagree, and the interleaved one is
correct.

**Checkpoint.**

<details>
<summary>1. You time an MPS matmul with <code>time.perf_counter()</code> and get 0.03 ms for work that takes 6 ms. What happened?</summary>

Accelerator work is dispatched asynchronously. The Python call returns as soon as the work
is queued, not when it completes, so the measurement records the cost of filling a queue.

`torch.utils.benchmark.Timer` avoids this because its default clock calls
`torch.accelerator.synchronize()` before reading the time. This is the single strongest
argument for using `Timer` rather than a hand-written loop.
</details>

<details>
<summary>2. You measure variant A three times, then variant B three times, and B appears 4% faster. What is wrong with this procedure?</summary>

Any drift in the machine's state during the cell — thermal behaviour, another process,
frequency scaling — is attributed entirely to B, because B was measured later. A 4% effect
is well within the drift observed over tens of seconds.

Interleaving the measurements round-robin (A, B, A, B, A, B) distributes the drift across
both variants. The notebook retains the flawed version alongside the corrected one, because
the flawed version produced a confidently wrong ordering.
</details>

<details>
<summary>3. <code>set_to_none=True</code> turns out to be about 5% faster. Is that the reason it is the default?</summary>

No. The time difference is small and would vanish on a noisier machine. The substantive
difference is in memory: `set_to_none=False` keeps a gradient tensor allocated for every
parameter, approximately 43 MB for this model, held for the entire run. `set_to_none=True`
releases them.

It is a memory optimisation that is commonly described as a speed optimisation. Measuring
it is how you find that out.
</details>

**Common confusion.** `timeit(n)` returns a single replicate and therefore no error bar.
`blocked_autorange(min_run_time=t)` chooses an iteration count automatically and returns
many replicates, which is what makes an IQR available. Prefer the latter unless you have a
specific reason not to.

---

## Self-assessment

You have understood the material if you can do the following without referring back.

- [ ] State what self time and total time each measure, and which to sort by for a given question.
- [ ] Name the flag combination required to make `export_stacks()` produce output.
- [ ] Explain why parameter count does not predict latency, using ResNet-18 as the example.
- [ ] Explain why FLOP count does not predict latency, using MobileNetV3-Small as the example.
- [ ] Classify a model as overhead-bound or compute-bound from one column of a profile.
- [ ] Explain why a forward pass retains memory when gradients are enabled.
- [ ] Describe an optimisation that improves inference and degrades training, and why.
- [ ] Capture a trace that HTA can read, and say what makes it readable.
- [ ] State your machine's benchmark noise floor, and how you measured it.
- [ ] Explain why measurements should be interleaved rather than grouped.
- [ ] Say which column of a trace diff is trustworthy, and why the other is not.

---

## Applying this to your own work

The closing exercise in both notebooks is the same, and it is the point of the workshops.

1. Take a model from your own project.
2. Benchmark it with `Timer.blocked_autorange()` and record the median and IQR.
3. Profile it, and classify the bottleneck using mean microseconds per operator.
4. Change exactly one thing.
5. Re-benchmark, and use `ab_compare()` to decide whether the change exceeds your noise floor.
6. If it does, capture traces before and after and run `TraceDiff.ops_diff` to identify
   which operators account for it.

Step 6 is the one most often skipped, and it is what converts a single result into
something you can apply to the next model. A number establishes that a change worked; a
diff establishes what kind of change works.

Then write three sentences: what was slow, what kind of slow it was, and what you changed.
If you cannot write the second sentence, return to step 3.
