"""Plot native 0.1.40 decoding policies from results.csv; requires matplotlib."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8", newline="")))
policies = [("nomtp", "No speculation", "#9caabe"),
            ("mtp", "MTP T4", "#087f8c"),
            ("ngram", "Ngram only", "#d28b35"),
            ("mtp-ngram", "MTP + ngram", "#765ca8")]
ink, muted, grid = "#162c43", "#506176", "#e4eaf0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                    "text.color": ink, "axes.labelcolor": muted,
                    "xtick.color": muted, "ytick.color": ink,
                    "axes.spines.top": False, "axes.spines.right": False,
                    "axes.spines.left": False, "axes.spines.bottom": False,
                    "svg.fonttype": "none", "svg.hashsalt": "rtxpro-v0140-ngram"})
fig = plt.figure(figsize=(14, 11.5), facecolor="#f7f9fc")
fig.text(.04, .961, "COMMUNITY MEASUREMENTS / UNMODIFIED STRATA 0.1.40",
         fontsize=10, fontweight="bold", color="#087f8c")
fig.text(.04, .914, "Four decoding paths on RTX PRO 6000", fontsize=27, fontweight="bold")
fig.text(.04, .877, "8K actual input + 512 output · One request · FP16 KV · 16K allocation",
         fontsize=13, color=muted)
gs = fig.add_gridspec(2, 2, left=.17, right=.962, bottom=.19, top=.78,
                      hspace=.85, wspace=.55)
for col, (quant, name) in enumerate([("q4", "Unsloth Q4_K_XL"), ("q8", "Unsloth Q8_0")]):
    data = {}
    for mode, _, _ in policies:
        matches = [r for r in rows if r["version"] == "v0.1.40" and
                   r["quant"] == quant and r["mode"] == mode]
        assert len(matches) == 1 and matches[0]["status"] == "completed"
        data[mode] = matches[0]
    for row, (metric, title, unit) in enumerate([
            ("decode_tok_s", "Streaming decode · higher is faster", "Committed output tokens / second"),
            ("total_seconds", "Whole request · lower is faster", "Seconds, including prefill")]):
        ax = fig.add_subplot(gs[row, col], facecolor="white")
        ax.set_title(name + "\n" + title, loc="left", fontsize=13,
                     fontweight="bold", pad=16, linespacing=1.5)
        values = [float(data[mode][metric]) for mode, _, _ in policies]
        limit = max(values)*1.22
        for i, ((mode, label, color), value) in enumerate(zip(policies, values)):
            ax.barh(i, value, height=.52, color=color, zorder=3)
            ax.text(value+limit*.02, i, f"{value:.1f}" if row == 0 else f"{value:.2f}",
                    va="center", fontsize=11)
        ax.set_yticks(range(4), [label for _, label, _ in policies])
        ax.set_ylim(3.6, -.6)
        ax.set_xlim(0, limit)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=grid, linewidth=.8)
        ax.set_xlabel(unit, fontsize=10, labelpad=9)
fig.text(.04, .104, "RTX PRO 6000 Blackwell Workstation 96 GB / 400 W · Ryzen 9 7950X · 128 GB RAM",
         fontsize=10, fontweight="bold")
fig.text(.04, .076, "Native 0.1.40 engine; one observation per path. Model loading excluded; existing compatibility packs.",
         fontsize=9.5, color=muted)
fig.text(.04, .048, "Ngram: suffix match ≥3, verifier up to T8. Combined: MTP T4 + suffix lookup + up to 4 chained lookup drafts.",
         fontsize=9.5, color=muted)
fig.text(.04, .021, "Q4 token streams matched. Q8 ngram paths first differ at output token 4; cause unresolved. Draft counts are in the report.",
         fontsize=9.5, color=muted)
for ext in ["png", "svg"]:
    fig.savefig(ROOT / f"ngram.{ext}", dpi=180, facecolor=fig.get_facecolor(),
                metadata={"Date": None} if ext == "svg" else None)
svg = ROOT / "ngram.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
