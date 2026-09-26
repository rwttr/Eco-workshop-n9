# PyTorch Performance Workshops

**241-353 AI Ecosystem · Prince of Songkla University**

Two self-contained notebooks on performance analysis in PyTorch. All material runs on a
laptop CPU and requires no NVIDIA GPU, no dataset downloads and no pretrained weights.

| Notebook | Tool | Question addressed | Runtime | Details |
|---|---|---|---|---|
| `pytorch_profiler_workshop.ipynb` | `torch.profiler` | **Where** does execution time go? | ~90 s | [documentation](docs/profiler-notebook.md) |
| `pytorch_benchmark_workshop.ipynb` | `torch.utils.benchmark` | **How much** time is there, and is a change real? | ~90 s | [documentation](docs/benchmark-notebook.md) |

The notebooks are intended to be taught in that order and applied as an iterative cycle:

> **benchmark to detect → profile to diagnose → change one thing → benchmark to confirm
> → trace-diff to explain**

Each tool answers a question the others cannot. A profiler total is not a latency figure,
since it includes tracing overhead. A benchmark result establishes magnitude but not cause.
A trace diff identifies which operators changed, which is what makes a result transferable
to the next model.

Both notebooks conclude with [trace analysis](docs/trace-analysis.md): Chrome Traces for
[Perfetto](https://ui.perfetto.dev), and Meta's
[Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis).

Working through the material without an instructor is the expected case. The
**[self-study guide](docs/self-study-guide.md)** supplies what a taught session otherwise
would: a session-by-session plan, the specific figures to examine in each output, and
checkpoint questions to answer before moving on.

---

## Setup

Python 3.10–3.13 is required; PyTorch does not yet publish wheels for 3.14.

```bash
# create the environment (any location; this is the path used during authoring)
python3.13 -m venv ~/ws_eco/.venv
source ~/ws_eco/.venv/bin/activate

# install
cd "path/to/ws_eco"
pip install -r requirements.txt

# run
jupyter lab pytorch_profiler_workshop.ipynb
```

Confirm the kernel selection under `Kernel ▸ Change Kernel`; it must point at the
environment created above. If the environment is not listed, register it:

```bash
python -m ipykernel install --user --name ws_eco --display-name "Python (ws_eco)"
```

The notebooks are then run from top to bottom.

Three dependencies are optional, and every cell that uses them degrades cleanly when they
are absent: `HolisticTraceAnalysis` (trace diffing), `ninja` (the C++ timer demonstration)
and `playwright` (regenerating the screenshots).

---

## The four models

The selection deliberately excludes U-Net and YOLO. Each model was chosen because it
defeats a *different* naive prediction of performance, and both notebooks use the same four,
so a result obtained in one applies directly to the other.

| Model | Family | Parameters | Defeats the prediction that… |
|---|---|---|---|
| **ResNet-18** | CNN (2015), `torchvision` | 11.7 M | — (the well-behaved baseline) |
| **MobileNetV3-Small** | Efficiency-designed CNN (2019), `torchvision` | 2.5 M | …fewer FLOPs means less time |
| **ViT-Tiny** | Vision Transformer (2020), implemented here | 2.9 M | …attention is inherently expensive |
| **VisionMamba-Tiny** | Selective State-Space Model (2024), implemented here | 0.16 M | …fewer parameters means faster |

Two results anchor the workshop. **MobileNetV3-Small performs roughly 32× less arithmetic
than ResNet-18 on the same input and takes about 2.5× longer**, because its depthwise
convolutions have no fused CPU kernel and decompose into hundreds of single-channel calls.
**VisionMamba-Tiny has approximately 70× fewer parameters than ResNet-18 and operates on
12× fewer pixels, yet takes roughly twice as long as ViT-Tiny.**

Neither parameter count nor FLOP count predicts either result. The profiler accounts for
both in a single column. Full architecture notes for all four are in
[docs/models.md](docs/models.md).

---

## Documentation

| Document | Contents |
|---|---|
| [docs/models.md](docs/models.md) | The four architectures: diagrams, measured comparison, what each one demonstrates, and where its code lives |
| [docs/profiler-notebook.md](docs/profiler-notebook.md) | Notebook 1: part list, profiler features covered, the architecture reference, and nine reproducible findings |
| [docs/benchmark-notebook.md](docs/benchmark-notebook.md) | Notebook 2: part list, coverage of `Timer`, `Measurement`, `Compare` and `Fuzzer`, and eight reproducible findings |
| [docs/trace-analysis.md](docs/trace-analysis.md) | Perfetto and HTA, with screenshots; two requirements that are easy to overlook; the scope of HTA on CPU traces; the dependency-free alternative |
| [docs/self-study-guide.md](docs/self-study-guide.md) | A follow-through manual for working through both notebooks unaided: a six-session plan, what to look for in each output, and checkpoint questions with worked answers |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Each error message these notebooks can produce, with an explanation |

---

## Files

| File | Purpose |
|---|---|
| `pytorch_profiler_workshop.ipynb` | Notebook 1, with outputs cleared for participants to run. |
| `pytorch_benchmark_workshop.ipynb` | Notebook 2, with outputs cleared. |
| `*_executed.ipynb` | Reference copies retaining all outputs, for preparation and for participants unable to install the environment. |
| `ws_models.py` | Shared model definitions (TinyViT, TinyVisionMamba, S6Naive/S6Fast). Imported by notebook 2; notebook 1 defines them inline, as their construction is part of that lesson. |
| `ws_trace.py` | Shared trace helpers: `capture_trace()`, `chrome_trace_ops()`, `add_rank()`, `quiet_hta()`, `HTA_ON_CPU`. |
| `make_screenshots.py` | Regenerates the screenshots in `docs/` using a headless browser. |
| `docs/` | The documents listed above, together with their images. |
| `requirements.txt` | Dependencies, with optional entries marked. |

---

## Course

Prepared for **241-353 AI Ecosystem**, Prince of Songkla University, September 2026.
Teaching material; reuse and adaptation for other courses is welcome.

---

## References

- [PyTorch Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) · [`torch.profiler` API](https://docs.pytorch.org/docs/stable/profiler.html)
- [PyTorch Benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) · [`torch.utils.benchmark` API](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
- [Perfetto UI](https://ui.perfetto.dev) · [Holistic Trace Analysis](https://github.com/facebookresearch/HolisticTraceAnalysis) · [FlameGraph](https://github.com/brendangregg/FlameGraph)
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (2020) — ViT
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)
