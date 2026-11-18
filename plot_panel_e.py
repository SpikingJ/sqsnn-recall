"""Paper panel (e): test accuracy against the silent delay d, with direct labels instead of a legend.

Same data and styling as plot_results.py (mean +/- s.d. over seeds, dots: seeds, shaded: delays not seen
in training, dashed: chance).  Every series is named in an empty region next to its own line.  After the
layout, the script measures, in points, the clearance between every label and every other drawn element
(lines, markers, error bars, seed dots, other labels) and the distance from each label to its own line.
Usage: python plot_panel_e.py --results results --out figures/panel_e
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

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--results", default=os.path.join(HERE, "results"))
p.add_argument("--tags", default="main,main_m5")
p.add_argument("--out", default=os.path.join(HERE, "figures", "panel_e"))
p.add_argument("--width", type=float, default=2.8)
p.add_argument("--height", type=float, default=2.1)
p.add_argument("--panel", default="(e)")
p.add_argument("--fs", type=float, default=6.2)
p.add_argument("--debug", action="store_true", help="also save <out>_debug.png with the label boxes")
p.add_argument("--label", action="append", default=[],
               help="override a label, e.g. --label 'sqs_chen=SQSNN (original ansatz)'")
a = p.parse_args()

# key: colour, marker, label, label centre (x = delay index, y = accuracy %), horizontal alignment
SERIES = {
    "sqs_event_m5": ("#1c5cab", "o", r"SQSNN ($N$ = 5)", (6.0, 79.0), "center"),
    "sqs_event":    ("#86b6ef", "h", r"SQSNN ($N$ = 1)", (0.62, 38.0), "center"),
    "lif":          ("#eb6834", "s", r"SNN ($\beta$ = 0.95)", (3.62, 31.0), "right"),
    "lif_lb":       ("#1baf7a", "P", "SNN\n" + r"(trainable $\beta$)", (6.5, 30.0), "center"),
    "qlif":         ("#eda100", "^", "QLIF-SNN", (1.18, 19.9), "center"),
    "sqs_chen":     ("#008300", "v", "SQSNN (original ansatz [2])", (2.4, 4.2), "center"),
}
for kv in a.label:
    k, v = kv.split("=", 1)
    SERIES[k] = SERIES[k][:2] + (v,) + SERIES[k][3:]
ORDER = ["lif", "lif_lb", "qlif", "sqs_chen", "sqs_event", "sqs_event_m5"]   # drawn last = on top
LW, MS, ELW, DOT_S = 1.3, 4.0, 0.7, 4

acc = defaultdict(lambda: defaultdict(list))
train_delays = set()
for f in glob.glob(os.path.join(a.results, "*.json")):
    r = json.load(open(f))
    if "test" not in r or r["args"]["tag"] not in a.tags.split(","):
        continue
    m = r["args"]["model"]
    if m == "sqs_event" and r["args"].get("n_mem", 1) > 1:
        m = f"sqs_event_m{r['args']['n_mem']}"
    if m not in SERIES:
        continue
    for d, v in r["test"].items():
        acc[m][int(d)].append(100 * v["acc"])
    train_delays |= set(r["train_delays"])

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b"})
fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=300)
all_d = sorted({d for m in acc for d in acc[m]})
pos = {d: i for i, d in enumerate(all_d)}
unseen = [d for d in all_d if d not in train_delays]
x0 = pos[unseen[0]] - 0.5
ax.axvspan(x0, len(all_d) - 0.5, color="#f0efec", lw=0, zorder=0)
note = ax.text(x0 + 0.08, 97, "not seen in\ntraining", ha="left", va="top", fontsize=6, color="#52514e")
ax.axhline(10, color="#898781", lw=0.7, ls=(0, (3, 2)), zorder=1)
geom, labels = {}, {}
for k, m in enumerate(ORDER):
    color, marker, label, (tx, ty), ha = SERIES[m]
    ds = sorted(acc[m])
    xs = np.array([pos[d] for d in ds]) + (k - (len(ORDER) - 1) / 2) * 0.07
    mu = np.array([np.mean(acc[m][d]) for d in ds])
    sd = np.array([np.std(acc[m][d]) for d in ds])
    for x, d in zip(xs, ds):
        ax.scatter([x] * len(acc[m][d]), acc[m][d], s=DOT_S, color=color, alpha=0.35, lw=0, zorder=3)
    ax.errorbar(xs, mu, yerr=sd, color=color, lw=LW, marker=marker, ms=MS, mew=0.6, mec="white",
                capsize=1.6, elinewidth=ELW, zorder=5 if m.startswith("sqs_event") else 4)
    labels[m] = ax.text(tx, ty, label, ha=ha, va="center", fontsize=a.fs, color="#0b0b0b", zorder=6,
                        linespacing=1.0, multialignment="center")
    geom[m] = (xs, mu, sd, [acc[m][d] for d in ds])
ax.set_xticks(range(len(all_d)))
ax.set_xticklabels([str(d) for d in all_d])
ax.set_xlim(-0.5, len(all_d) - 0.5)
ax.set_xlabel("Silent delay $d$ before the go cue (steps of 25 ms)")
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e1e0d9", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.95))   # room above the axes for the panel label
if a.panel:
    fig.text(0.005, 0.995, a.panel, ha="left", va="top", fontsize=9, fontweight="bold")

# ---------------------------------------------------------------- clearance check (in points)
fig.canvas.draw()
R = fig.canvas.get_renderer()
pt = 72.0 / fig.dpi                                   # points per pixel
to_px = ax.transData.transform


def elements(m):
    """Named drawn elements of series m as (name, points in px, half-thickness in pt)."""
    xs, mu, sd, seeds = geom[m]
    ds = sorted(acc[m])
    out = []
    t = np.linspace(0, 1, 120)
    for i in range(len(xs) - 1):
        seg = np.c_[xs[i] + t * (xs[i + 1] - xs[i]), mu[i] + t * (mu[i + 1] - mu[i])]
        out.append((f"line {ds[i]}-{ds[i + 1]}", to_px(seg), LW / 2))
    for x, u, s_, sv, d in zip(xs, mu, sd, seeds, ds):
        out.append((f"marker {d}", to_px(np.array([[x, u]])), MS / 2 + 0.3))
        yy = np.linspace(u - s_, u + s_, 40)
        out.append((f"error bar {d}", to_px(np.c_[np.full_like(yy, x), yy]), ELW / 2))
        for yc in (u - s_, u + s_):
            c = to_px(np.array([[x, yc]]))[0]
            cap = np.c_[c[0] + np.linspace(-1.6, 1.6, 15) / pt, np.full(15, c[1])]
            out.append((f"cap {d}", cap, 0.5))
        out.append((f"seeds {d}", to_px(np.c_[np.full(len(sv), x), sv]), np.sqrt(DOT_S / np.pi)))
    return out


def bbox_px(t):
    b = t.get_window_extent(R)
    return np.array([b.x0, b.y0, b.x1, b.y1])


def clearance(box, pts_px, radius_pt):
    x0_, y0_, x1_, y1_ = box
    dx = np.maximum(np.maximum(x0_ - pts_px[:, 0], pts_px[:, 0] - x1_), 0)
    dy = np.maximum(np.maximum(y0_ - pts_px[:, 1], pts_px[:, 1] - y1_), 0)
    return float(np.min(np.hypot(dx, dy)) * pt - radius_pt)


def box_gap(b1, b2):
    dx = max(b2[0] - b1[2], b1[0] - b2[2], 0)
    dy = max(b2[1] - b1[3], b1[1] - b2[3], 0)
    inside = b2[0] < b1[2] and b1[0] < b2[2] and b2[1] < b1[3] and b1[1] < b2[3]
    return -1.0 if inside else float(np.hypot(dx, dy) * pt)


inv = ax.transData.inverted()
frame = ax.get_window_extent(R)
chance = to_px(np.c_[np.linspace(-0.5, len(all_d) - 0.5, 600), np.full(600, 10.0)])
worst, report = np.inf, []
for m in ORDER:
    box = bbox_px(labels[m])
    own = [(clearance(box, q, r), n) for n, q, r in elements(m)]
    own_line = min(c for c, n in own if n.startswith("line"))
    near = [(c, f"own {n}") for c, n in own if not n.startswith("line")]
    for m2 in ORDER:
        if m2 != m:
            near += [(clearance(box, q, r), f"{m2} {n}") for n, q, r in elements(m2)]
            near.append((box_gap(box, bbox_px(labels[m2])), f"label {m2}"))
    near.append((clearance(box, chance, 0.35), "chance line"))
    near.append((box_gap(box, bbox_px(note)), "note"))
    near.append((min(box[0] - frame.x0, frame.x1 - box[2], box[1] - frame.y0, frame.y1 - box[3]) * pt, "axes frame"))
    near.sort()
    (dx0, dy0), (dx1, dy1) = inv.transform(box[:2]), inv.transform(box[2:])
    print(f"{m:13s} box x {dx0:5.2f}..{dx1:5.2f} y {dy0:5.1f}..{dy1:5.1f} | own line {own_line:4.1f} pt | "
          + ", ".join(f"{c:4.1f} {n}" for c, n in near[:3]))
    worst = min(worst, near[0][0])
print("OK: every label clears all other elements by at least 1 pt" if worst >= 1.0
      else f"CHECK: smallest clearance {worst:.1f} pt")
if a.debug:
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import IdentityTransform
    for m in ORDER:
        b = bbox_px(labels[m])
        fig.add_artist(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], transform=IdentityTransform(), fill=False,
                                 ec="red", lw=0.3))
    fig.savefig(f"{a.out}_debug.png")
    for art in fig.artists[-len(ORDER):]:
        art.remove()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}")
print("saved", a.out)
