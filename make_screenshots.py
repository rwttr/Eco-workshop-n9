#!/usr/bin/env python3
"""Regenerate the screenshots used in README.md and docs/trace-analysis.md.

Captures real screenshots - nothing here is a mock-up:

  docs/perfetto_overview.png   ui.perfetto.dev with a workshop trace loaded
  docs/perfetto_zoom.png       the same trace zoomed into one ProfilerStep
  docs/hta_analysis.png        Holistic Trace Analysis output, rendered from the
                               executed profiler notebook
  docs/chrome_trace_table.png  the dependency-free Chrome-Trace parser's output
  docs/profiler_table.png      a torch.profiler key_averages() table, from the
                               executed profiler notebook
  docs/benchmark_compare.png   a torch.utils.benchmark Compare table, from the
                               executed benchmark notebook

Run the notebooks first (they write the traces this script needs), then:

    pip install playwright && python -m playwright install chromium
    python make_screenshots.py

Perfetto is loaded by driving its file picker in a headless browser. The trace is sent to
ui.perfetto.dev the same way a drag-and-drop would send it: the page processes it locally
in the browser, it is not uploaded to a server.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"
PERFETTO = "https://ui.perfetto.dev/"


def _need(module, hint):
    try:
        __import__(module)
    except ImportError:
        sys.exit(f"{module} is required. {hint}")


def find_trace():
    """Prefer the profiler notebook's baseline training trace."""
    for pattern in ["profiler_out/traces/baseline/*.pt.trace.json",
                    "profiler_out/traces/*/*.pt.trace.json",
                    "benchmark_out/traces/*/*.pt.trace.json"]:
        hits = sorted(ROOT.glob(pattern))
        if hits:
            return hits[0]
    sys.exit("No trace found. Run pytorch_profiler_workshop.ipynb first.")


def shoot_perfetto(trace):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 620},
                                device_scale_factor=2)
        page.goto(PERFETTO, wait_until="networkidle")
        page.wait_for_timeout(2500)
        try:                                            # cookie banner
            page.get_by_text("OK", exact=True).first.click()
        except Exception:
            pass
        page.wait_for_timeout(500)

        page.locator("input[type=file]").first.set_input_files(str(trace))
        for _ in range(90):
            page.wait_for_timeout(1000)
            if "Spans" in page.inner_text("body"):
                break
        page.wait_for_timeout(2000)

        for pattern in ["Process ", "Spans "]:          # expand the track groups
            try:
                page.locator(f"text=/{pattern}\\d+/").first.click()
                page.wait_for_timeout(1200)
            except Exception:
                pass
        page.wait_for_timeout(2500)

        clip = {"x": 0, "y": 0, "width": 1500, "height": 400}
        page.screenshot(path=str(DOCS / "perfetto_overview.png"), clip=clip)
        print("  wrote docs/perfetto_overview.png")

        page.mouse.move(700, 250)                       # zoom into step 1
        for _ in range(9):
            page.keyboard.press("w")
            page.wait_for_timeout(350)
        page.wait_for_timeout(1500)
        page.mouse.move(820, 285)                       # hover an op for its tooltip
        page.wait_for_timeout(1500)
        page.screenshot(path=str(DOCS / "perfetto_zoom.png"), clip=clip)
        print("  wrote docs/perfetto_zoom.png")
        browser.close()


def shoot_notebook_cells(notebook, out_png, keys, title, width=1250):
    """Render selected executed cells to HTML, then screenshot them."""
    import nbformat
    from playwright.sync_api import sync_playwright

    nb = nbformat.read(notebook, as_version=4)
    picked = [c for c in nb.cells
              if c.cell_type == "code" and any(k in c.source for k in keys)]
    for c in picked:                                    # drop libtorch's USDT stderr noise
        c.outputs = [o for o in c.outputs if o.get("name") != "stderr"]
    if not picked:
        print(f"  ! no matching cells in {notebook.name}; skipping {out_png.name}")
        return

    sub = nbformat.v4.new_notebook()
    sub.cells = [nbformat.v4.new_markdown_cell(title)] + picked
    sub.metadata = nb.metadata

    tmp = Path(tempfile.mkdtemp())
    ipynb = tmp / "sub.ipynb"
    nbformat.write(sub, ipynb)
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
                    "--template", "lab", "--log-level=ERROR", str(ipynb)], check=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": 200},   # full_page
                                device_scale_factor=2)                    # grows to fit
        page.goto(ipynb.with_suffix(".html").as_uri(), wait_until="networkidle")
        page.wait_for_timeout(2000)
        page.screenshot(path=str(out_png), full_page=True)
        print(f"  wrote docs/{out_png.name}")
        browser.close()


def main():
    _need("playwright", "pip install playwright && python -m playwright install chromium")
    _need("nbformat", "pip install nbformat")
    DOCS.mkdir(exist_ok=True)

    trace = find_trace()
    print(f"trace: {trace.relative_to(ROOT)} ({trace.stat().st_size / 1e6:.1f} MB)")

    print("Perfetto:")
    shoot_perfetto(trace)

    nb = ROOT / "pytorch_profiler_workshop_executed.ipynb"
    if not nb.exists():
        nb = ROOT / "pytorch_profiler_workshop.ipynb"

    print("Holistic Trace Analysis:")
    shoot_notebook_cells(
        nb, DOCS / "hta_analysis.png",
        keys=["TraceAnalysis(trace_dir=str(base_dir))", "symbols = analyzer",
              "per_iter", "ops_diff(analyzer"],
        title="## Holistic Trace Analysis on a PyTorch CPU trace\n"
              "*cells from `pytorch_profiler_workshop.ipynb`, Part 5*")

    print("Chrome Trace, parsed directly:")
    shoot_notebook_cells(
        nb, DOCS / "chrome_trace_table.png",
        keys=["ops = chrome_trace_ops(base_file", "ref.merge"],
        title="## The same trace, parsed with `json` + `pandas` only\n"
              "*cells from `pytorch_profiler_workshop.ipynb`, Part 5.2*")

    print("torch.profiler table:")
    shoot_notebook_cells(
        nb, DOCS / "profiler_table.png",
        keys=['key_averages().table(sort_by="self_cpu_time_total", row_limit=8)'],
        title="## `torch.profiler`: where the time goes\n"
              "*cell from `pytorch_profiler_workshop.ipynb`, Part 1.1*")

    bnb = ROOT / "pytorch_benchmark_workshop_executed.ipynb"
    if not bnb.exists():
        bnb = ROOT / "pytorch_benchmark_workshop.ipynb"

    print("torch.utils.benchmark Compare:")
    shoot_notebook_cells(
        bnb, DOCS / "benchmark_compare.png",
        keys=['label="Forward pass, batch 4"'],
        title="## `torch.utils.benchmark`: how much time, with error bars\n"
              "*cell from `pytorch_benchmark_workshop.ipynb`, Part 5*")

    print("\nDone. Screenshots in docs/")


if __name__ == "__main__":
    main()
