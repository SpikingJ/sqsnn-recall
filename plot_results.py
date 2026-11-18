"""Figure: test accuracy against the delay between the end of the event stream and the go cue.

Reads results/<tag>_<model>_*.json written by train.py and draws the mean over seeds with +/- 1 s.d.
error bars (dots: individual seeds).  Delays to the right of the shaded boundary were never seen in
training.  Colours: slots 1-6 of the dataviz reference categorical palette, in its fixed order; every
series also has its own marker, so identity never rests on colour alone.
Usage: python plot_results.py --tag main --out figures/delay_accuracy
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--tag", default="main", help="comma-separated run tags to include")
p.add_argument("--results", default=os.path.join(HERE, "results"))
p.add_argument("--out", default=os.path.join(HERE, "figures", "delay_accuracy"))
p.add_argument("--step_ms", type=float, default=25.0)
p.add_argument("--models", default="sqs_event,lif,lif_lb,qlif,sqs_read,sqs_chen")
p.add_argument("--csv", default="", help="also write the plotted numbers to this CSV file")
p.add_argument("--label", action="append", default=[], help="override a legend label, e.g. sqs_event_m2='SQS, proposed'")
p.add_argument("--color", action="append", default=[], help="override a colour, e.g. sqs_event='#6da7ec'")
p.add_argument("--marker", action="append", default=[], help="override a marker, e.g. sqs_event=h")
p.add_argument("--ncol", type=int, default=2)
p.add_argument("--width", type=float, default=3.45, help="figure width in inches (3.45 = one IEEE column)")
p.add_argument("--height", type=float, default=0.0, help="figure height in inches (0 = from the legend size)")
p.add_argument("--panel", default="", help="panel label drawn at the top left, e.g. (e)")
p.add_argument("--legend_fs", type=float, default=6.0)
a = p.parse_args()

STYLE = {   # label, colour (reference slots 1..6 in order), marker
    # proposed ansatz: one blue ramp (reference steps 250/350/450/550), darker = more memory qubits
    "sqs_event": ("SQS, proposed, 1 memory qubit", "#86b6ef", "h"),
    "sqs_event_m2": ("SQS, proposed, 2 memory qubits", "#6da7ec", "H"),
    "sqs_event_m3": ("SQS, proposed, 3 memory qubits", "#5598e7", "p"),
    "sqs_event_m4": ("SQS, proposed, 4 memory qubits", "#2a78d6", "8"),
    "sqs_event_m5": ("SQS, proposed, 5 memory qubits", "#1c5cab", "o"),
    "lif":       (r"LIF, fixed leak ($\beta$ = 0.95)", "#eb6834", "s"),
    "lif_lb":    ("LIF, trained leak", "#1baf7a", "P"),
    "if":        ("IF (no leak), with bias", "#6250d6", "X"),
    "if_nb":     ("IF (no leak), no bias", "#e34948", "*"),
    "qlif":      ("QLIF", "#eda100", "^"),
    "sqs_read":  ("SQS, read gate", "#e87ba4", "D"),
    "sqs_chen":  ("SQS, original ansatz", "#008300", "v"),
}

for kind, items in (("label", a.label), ("color", a.color), ("marker", a.marker)):
    for kv in items:
        k, v = kv.split("=", 1)
        lab, col, mk = STYLE[k]
        STYLE[k] = {"label": (v, col, mk), "color": (lab, v, mk), "marker": (lab, col, v)}[kind]

acc = defaultdict(lambda: defaultdict(list))
params, train_delays = {}, set()
tags = a.tag.split(",")
for f in glob.glob(os.path.join(a.results, "*.json")):
    r = json.load(open(f))
    if "test" not in r or r["args"]["tag"] not in tags:
        continue
    m = r["args"]["model"]
    if m == "lif" and r["args"].get("beta", 0.95) == 1.0:
        m = "if" if r["args"].get("bias", 1) else "if_nb"
    if m == "sqs_event" and r["args"].get("n_mem", 1) > 1:
        m = f"sqs_event_m{r['args']['n_mem']}"
    for d, v in r["test"].items():
        acc[m][int(d)].append(100 * v["acc"])
    params[m] = r["params"]
    train_delays |= set(r["train_delays"])

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
fig, ax = plt.subplots(figsize=(3.45, 2.45), dpi=300)
models = [m for m in a.models.split(",") if m in acc]
all_d = sorted({d for m in models for d in acc[m]})
pos = {d: i for i, d in enumerate(all_d)}             # categorical x positions (delays double)
unseen = [d for d in all_d if d not in train_delays]
if unseen:
    x0 = pos[unseen[0]] - 0.5
    ax.axvspan(x0, len(all_d) - 0.5, color="#f0efec", lw=0, zorder=0)
    ax.text(x0 + 0.08, 97, "not seen in\ntraining", ha="left", va="top", fontsize=6, color="#52514e")
ax.axhline(10, color="#898781", lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.text(-0.45, 11.0, "chance", ha="left", va="bottom", fontsize=6, color="#52514e")
rows = []
for k, m in enumerate(models):
    label, color, marker = STYLE[m]
    ds = sorted(acc[m])
    xs = np.array([pos[d] for d in ds]) + (k - (len(models) - 1) / 2) * min(0.07, 0.45 / len(models))
    mu = np.array([np.mean(acc[m][d]) for d in ds])
    sd = np.array([np.std(acc[m][d]) for d in ds])
    for x, d in zip(xs, ds):
        ax.scatter([x] * len(acc[m][d]), acc[m][d], s=4, color=color, alpha=0.35, lw=0, zorder=3)
    ax.errorbar(xs, mu, yerr=sd, color=color, lw=1.3, marker=marker, ms=4.0, mew=0.6, mec="white",
                capsize=1.6, elinewidth=0.7, label=label, zorder=5 if m.startswith("sqs_event") else 4)
    print(f"{m:10s} params={params[m]:7d} n_seeds={len(acc[m][ds[0]])} " +
          "  ".join(f"d={d}: {u:.1f}+-{s:.1f}" for d, u, s in zip(ds, mu, sd)))
    rows += [(m, params[m], d, len(acc[m][d]), u, s) for d, u, s in zip(ds, mu, sd)]
ax.set_xticks(range(len(all_d)))
ax.set_xticklabels([str(d) for d in all_d])
ax.set_xlim(-0.5, len(all_d) - 0.5)
ax.set_xlabel(f"Silent delay $d$ before the go cue (steps of {a.step_ms:.0f} ms)")
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e1e0d9", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
handles = [Line2D([0], [0], color=STYLE[m][1], marker=STYLE[m][2], lw=1.3, ms=4.0, mec="white", mew=0.6)
           for m in models]
nrow = (len(models) + a.ncol - 1) // a.ncol
ax.legend(handles, [STYLE[m][0] for m in models], frameon=False, fontsize=a.legend_fs, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), ncol=a.ncol, handlelength=1.8, columnspacing=1.0, handletextpad=0.4)
fig.set_size_inches(a.width, a.height if a.height > 0 else 2.05 + 0.13 * nrow)
fig.tight_layout(pad=0.3)
if a.panel:
    fig.text(0.005, 0.995, a.panel, ha="left", va="top", fontsize=9, fontweight="bold")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}")
if a.csv:
    with open(a.csv, "w") as f:
        f.write("model,params,delay_steps,n_seeds,test_acc_mean,test_acc_sd\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.2f},{r[5]:.2f}\n")
print("saved", a.out + ".png/.pdf")
