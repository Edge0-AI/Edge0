"""Render the four measured configuration points from results.csv."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8", newline="")))
ink, muted, grid = "#162c43", "#506176", "#e4eaf0"
colors = {"chunk1024": "#9caabe", "chunk8192": "#087f8c"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                    "text.color": ink, "axes.labelcolor": muted,
                    "xtick.color": muted, "ytick.color": ink,
                    "axes.spines.top": False, "axes.spines.right": False,
                    "axes.spines.left": False, "axes.spines.bottom": False,
                    "svg.fonttype": "none", "svg.hashsalt": "q8-prefill8192"})
fig = plt.figure(figsize=(14, 7.8), facecolor="#f7f9fc")
fig.text(.04, .954, "NATIVE STRATA 0.1.40 / Q8_0 CONFIGURATION EXPERIMENT",
         fontsize=10, fontweight="bold", color=colors["chunk8192"])
fig.text(.04, .901, "6.7× faster Q8 prefill", fontsize=31, fontweight="bold")
fig.text(.04, .862, "1,024 → 8,192-token chunks · Same model and expert cache · FP16 KV · MTP T4",
         fontsize=13, color=muted)
fig.legend(handles=[Patch(color=colors["chunk1024"], label="Chunk 1,024"),
                    Patch(color=colors["chunk8192"], label="Chunk 8,192")],
           loc="upper right", bbox_to_anchor=(.97, .96), frameon=False, ncol=2)
gs = fig.add_gridspec(1, 2, left=.15, right=.962, top=.69, bottom=.25, wspace=.35)
for col, (metric, title) in enumerate([("prefill_seconds", "Prefill time"),
                                     ("total_seconds", "Whole request, including prefill")]):
    ax = fig.add_subplot(gs[0, col], facecolor="white")
    ax.set_title(title + " · lower is faster", loc="left", fontsize=13, fontweight="bold", pad=20)
    for i, tokens in enumerate([32768, 131072]):
        for variant, offset in [("chunk1024", -.20), ("chunk8192", .20)]:
            match = [r for r in rows if int(r["input_tokens"]) == tokens and r["variant"] == variant]
            assert len(match) == 1
            value = float(match[0][metric])
            ax.barh(i+offset, value, height=.32, color=colors[variant], zorder=3)
            ax.text(value+4, i+offset, f"{value:.2f} s", va="center", fontsize=11)
    ax.set_yticks([0, 1], ["32K input", "128K input"])
    ax.set_ylim(1.6, -.6)
    ax.set_xlim(0, 275)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=grid, linewidth=.8)
    ax.set_xlabel("Seconds / request", labelpad=10)
fig.text(.04, .158, "RTX PRO 6000 Blackwell Workstation 96 GB / 400 W · Ryzen 9 7950X · 128 GB RAM",
         fontsize=10, fontweight="bold")
fig.text(.04, .113, "1,024 output tokens per request. Decode remained 143–147 tok/s with the larger chunk. Model loading excluded.",
         fontsize=10, color=muted)
fig.text(.04, .073, "Unmodified engine and weights; prefill borrowing disabled. Single observations; output-token differences are recorded.",
         fontsize=9.5, color=muted)
fig.text(.04, .035, "Measured 2026-10-06. The faster configuration's hardware profile is in progress.", fontsize=9.5, color=muted)
for ext in ["png", "svg"]:
    fig.savefig(ROOT / f"overview.{ext}", dpi=180, facecolor=fig.get_facecolor(),
                metadata={"Date": None} if ext == "svg" else None)
svg = ROOT / "overview.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
