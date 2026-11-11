"""Render the recorded release comparison; requires matplotlib, no GPU."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8", newline="")))
groups = [("q4", "mtp", "Q4_K_XL / MTP T4"),
          ("q4", "nomtp", "Q4_K_XL / no MTP"),
          ("q8", "mtp", "Q8_0 / MTP T4"),
          ("q8", "nomtp", "Q8_0 / no MTP")]
colors = {"v0.1.39": "#9caabe", "v0.1.40": "#087f8c"}
ink, muted, grid = "#162c43", "#506176", "#e4eaf0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                    "text.color": ink, "axes.labelcolor": muted,
                    "xtick.color": muted, "ytick.color": ink,
                    "axes.spines.top": False, "axes.spines.right": False,
                    "axes.spines.left": False, "axes.spines.bottom": False,
                    "svg.fonttype": "none", "svg.hashsalt": "rtxpro-v0140"})
fig = plt.figure(figsize=(14, 8.4), facecolor="#f7f9fc")
fig.text(.04, .956, "COMMUNITY MEASUREMENTS / UNMODIFIED RELEASE ENGINES",
         fontsize=10, fontweight="bold", color=colors["v0.1.40"])
fig.text(.04, .905, "Strata 0.1.39 → 0.1.40", fontsize=29, fontweight="bold")
fig.text(.04, .867, "Q4 and Q8 on RTX PRO 6000 Blackwell · 8K input + 512 output · FP16 KV",
         fontsize=13, color=muted)
fig.text(.04, .809, "Q8 now loads natively", fontsize=20, fontweight="bold")
fig.text(.04, .780, "Stock 0.1.39 rejects its Q8 PLE table; no patched baseline is plotted.",
         fontsize=11, color=muted)
fig.text(.64, .809, "Q4 decode improves", fontsize=20, fontweight="bold",
         color=colors["v0.1.40"])
gains = []
for mode in ["mtp", "nomtp"]:
    values = {r["version"]: float(r["decode_tok_s"]) for r in rows
              if r["quant"] == "q4" and r["mode"] == mode}
    gains.append(100 * (values["v0.1.40"] / values["v0.1.39"] - 1))
fig.text(.64, .780, f"+{gains[0]:.1f}% MTP · +{gains[1]:.1f}% no MTP (single runs)",
         fontsize=11, color=muted)
fig.legend(handles=[Patch(color=colors[v], label=v) for v in colors],
           loc="upper right", bbox_to_anchor=(.97, .966), frameon=False, ncol=2)
gs = fig.add_gridspec(1, 2, left=.215, right=.963, bottom=.205, top=.690, wspace=.27)
for col, (metric, title, unit, limit) in enumerate([
        ("decode_tok_s", "Streaming decode · higher is faster", "Output tokens / second", 310),
        ("total_seconds", "Whole request · lower is faster", "Seconds, including prefill", 23)]):
    ax = fig.add_subplot(gs[0, col], facecolor="white")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=18)
    for i, (quant, mode, _) in enumerate(groups):
        for version, offset in [("v0.1.39", -.19), ("v0.1.40", .19)]:
            match = [r for r in rows if r["quant"] == quant and r["mode"] == mode
                     and r["version"] == version]
            assert len(match) == 1
            row = match[0]
            if row["status"] != "completed":
                ax.text(limit*.018, i+offset, "Unsupported Q8 PLE", va="center",
                        color=muted, fontsize=9.5, fontstyle="italic")
                continue
            value = float(row[metric])
            ax.barh(i+offset, value, height=.30, color=colors[version], zorder=3)
            label = f"{value:.1f}" if col == 0 else f"{value:.2f}"
            ax.text(value+limit*.018, i+offset, label, va="center", fontsize=10)
    ax.set_yticks(range(4), [g[2] for g in groups] if col == 0 else [""]*4)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_ylim(3.6, -.6)
    ax.set_xlim(0, limit)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=grid, linewidth=.8)
    ax.set_xlabel(unit, fontsize=10, labelpad=10)
fig.text(.04, .124, "RTX PRO 6000 Blackwell Workstation 96 GB / 400 W · Ryzen 9 7950X · 128 GB RAM",
         fontsize=10, fontweight="bold")
fig.text(.04, .092, "Single observation per configuration; model loading excluded. Existing compatibility packs used by both engines.",
         fontsize=9.5, color=muted)
fig.text(.04, .061, "16,384-token allocation · one request · temperature 0 · ngram off · each timed request reads all 8,192 input tokens.",
         fontsize=9.5, color=muted)
fig.text(.04, .030, "Release comparison, not a PLE-only ablation or an intelligence benchmark. Token checks and limitations are in the report.",
         fontsize=9.5, color=muted)
for ext in ["png", "svg"]:
    fig.savefig(ROOT / f"overview.{ext}", dpi=180, facecolor=fig.get_facecolor(),
                metadata={"Date": None} if ext == "svg" else None)
svg = ROOT / "overview.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
