# Draws the README figures as SVG.
#   cd dayahead && .venv/bin/python ../docs/make_figures.py
# docs/figures/battery_gap.svg : OASIS "actual" vs Outlook demand vs CAISO's forecast, by hour
# docs/figures/headline.svg    : test-period MAE vs CAISO, with 95% intervals
# docs/figures/by_hour.svg     : test-period MAE by hour of day
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dayahead"))
import data  # noqa: E402
import exp  # noqa: E402
from final import A_PARTS, B_EXTRA, bootstrap_gap  # noqa: E402

OUT = os.path.join(HERE, "figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, MUTED, GRID, AXIS = "#1f2328", "#57606a", "#e6e8eb", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "svg.fonttype": "none", "figure.facecolor": "white",
})


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=14, fontweight="bold", pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=11, color=MUTED, va="bottom")


def label_end(ax, x, y, s, color, dy=0):
    ax.annotate(s, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=11, color=INK, fontweight="bold")
    ax.plot([x], [y], "o", color=color, ms=5)


def battery_gap():
    h = data.hourly()
    d = h[(h.day >= "2026-04-01") & (h.day < "2026-10-01")]
    prof = d.groupby("hour")[["load_oasis", "load", "caiso_dam"]].mean() / 1000
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = prof.index
    ax.fill_between(x, prof.load, prof.load_oasis, color=AQUA, alpha=0.15, lw=0,
                    label="gap = batteries charging")
    ax.plot(x, prof.load_oasis, color=AQUA, lw=2, label='OASIS "actual load" (incl. battery charging)')
    ax.plot(x, prof.load, color=BLUE, lw=2, label="demand, Today's Outlook (our target)")
    ax.plot(x, prof.caiso_dam, color=ORANGE, lw=2, ls=(0, (4, 3)), label="CAISO's day-ahead forecast")
    gap = (prof.load_oasis - prof.load).idxmax()
    ax.annotate(f"≈{prof.load_oasis[gap] - prof.load[gap]:.1f} GW at {gap}:00",
                (gap, (prof.load_oasis[gap] + prof.load[gap]) / 2), ha="center", fontsize=11, color=INK)
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, handlelength=2.2)
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23], ["0:00", "6:00", "12:00", "18:00", "23:00"])
    ax.set_ylabel("GW")
    title(ax, "The forecast and the \"actual\" measure different things",
          "Average hourly load, Apr–Sep 2026, Pacific time")
    ax.set_ylim(21.5, 36.5)    # headroom for the legend
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.1)
    fig.savefig(os.path.join(OUT, "battery_gap.svg"))


def predictions():
    df = exp.table()
    P = {n: exp.run(n, "test") for n in A_PARTS + [B_EXTRA]}
    A = sum(P[n] for n in A_PARTS) / len(A_PARTS)
    c = df.caiso_dam.reindex(A.index)
    B = (c + A + P[B_EXTRA]) / 3
    return df.reindex(A.index), {"CAISO": c, "A": A, "B": B}


def headline(df, preds):
    y = df.load
    rows = [("CAISO day-ahead", preds["CAISO"], ORANGE),
            ("A: ours, no CAISO inputs", preds["A"], AQUA),
            ("B: ours + CAISO's forecast", preds["B"], BLUE)]
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for i, (name, p, color) in enumerate(rows):
        mae = (p - y).abs().mean()
        ax.barh(i, mae, height=0.55, color=color)
        note = f"{mae:.0f} MW"
        if name != "CAISO day-ahead":
            lo, hi = bootstrap_gap(y, p, preds["CAISO"], df.day)
            gap = mae - (preds["CAISO"] - y).abs().mean()
            note += f"  ·  {gap:+.0f} vs CAISO  [{lo:+.0f}, {hi:+.0f}]".replace("-", "−")
        ax.text(mae + 8, i, note, va="center", fontsize=11, color=INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.tick_params(axis="y", length=0, labelcolor=INK, labelsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 980)
    ax.set_xticks([0, 200, 400, 600])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean absolute error (MW), lower is better")
    title(ax, "Test period: Jan 2025 – Sep 2026", "15,311 hours, issued 9:00 the day before · [95% CI, bootstrap over days]")
    fig.subplots_adjust(left=0.28, right=0.97, top=0.78, bottom=0.2)
    fig.savefig(os.path.join(OUT, "headline.svg"))


def by_hour(df, preds):
    fig, ax = plt.subplots(figsize=(8, 4.0))
    for key, name, color in [("CAISO", "CAISO", ORANGE), ("A", "A", AQUA), ("B", "B", BLUE)]:
        e = (preds[key] - df.load).abs().groupby(df.hour).mean()
        ax.plot(e.index, e.values, color=color, lw=2, label={
            "CAISO": "CAISO day-ahead", "A": "A: ours, no CAISO inputs", "B": "B: ours + CAISO's forecast"}[key])
        label_end(ax, 23, e[23], name, color, {"CAISO": -2, "A": 10, "B": -8}[key])
    ax.set_xlim(0, 23)
    ax.set_ylim(0, None)
    ax.set_xticks([0, 6, 12, 18, 23], ["0:00", "6:00", "12:00", "18:00", "23:00"])
    ax.set_ylabel("MAE (MW)")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    title(ax, "Errors peak midday, when rooftop solar moves demand the most",
          "Test period, mean absolute error by hour of day")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.84, bottom=0.1)
    fig.savefig(os.path.join(OUT, "by_hour.svg"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    battery_gap()
    df, preds = predictions()
    headline(df, preds)
    by_hour(df, preds)
    print("wrote", sorted(os.listdir(OUT)))
