"""Task panel: delayed classification of MNIST-DVS (event frames, silent delay, go cue / response window).

Usage: python draw_task.py --out figures/task --panel "(d)"
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(HERE, "figures", "task"))
p.add_argument("--panel", default="")
p.add_argument("--width", type=float, default=2.95)
p.add_argument("--height", type=float, default=0.78)
p.add_argument("--sample", type=int, default=3)
a = p.parse_args()

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
INK, MUTED, ACC, ACC_FILL, SIL = "#0b0b0b", "#52514e", "#eb6834", "#fbe1d6", "#ececec"

d = np.load(os.path.join(HERE, "data", "mnist_dvs_T20.npz"))
x = np.unpackbits(d["x_test"], axis=-1)[..., :int(d["n_ch"])]
frames = x[a.sample].reshape(20, 26, 26)
imgs = [frames[0:7].max(0), frames[7:14].max(0), frames[14:20].max(0)]      # events of a few frames

fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=300)
ax.set_xlim(0, 11.3)
ax.set_ylim(-0.75, 3.05)
ax.axis("off")
bs, gap, y0 = 0.86, 0.1, 0.75


def brace(x0, x1, y, up, text, color):
    """Curly bracket from x0 to x1 at height y, opening up (text above) or down (text below)."""
    s = 1 if up else -1
    xs = np.linspace(x0, x1, 200)
    mid = (x0 + x1) / 2
    h = 0.16
    yy = y + s * h * (1 - np.exp(-((xs - x0) / 0.12) ** 2) * 0 - 0)  # flat part
    ax.plot([x0, x0], [y, y + s * h * 0.6], color=color, lw=0.8)
    ax.plot([x1, x1], [y, y + s * h * 0.6], color=color, lw=0.8)
    ax.plot([x0, mid - 0.12], [y + s * h * 0.6] * 2, color=color, lw=0.8)
    ax.plot([mid + 0.12, x1], [y + s * h * 0.6] * 2, color=color, lw=0.8)
    ax.plot([mid - 0.12, mid, mid + 0.12], [y + s * h * 0.6, y + s * h * 1.3, y + s * h * 0.6], color=color, lw=0.8)
    ax.text(mid, y + s * h * 1.6, text, ha="center", va="bottom" if up else "top", fontsize=7.0, color=color)


x = 0.25
starts = {}
# event frames
starts["frames"] = x
for k, img in enumerate(imgs):
    ax.imshow(1 - img.T, cmap="gray", vmin=0, vmax=1, extent=(x, x + bs, y0, y0 + bs), zorder=2,
              interpolation="nearest")
    ax.add_patch(Rectangle((x, y0), bs, bs, fc="none", ec=INK, lw=0.7, zorder=3))
    x += bs + gap
    if k == 1:
        ax.text(x + 0.12, y0 + bs / 2, r"$\cdots$", ha="center", va="center", fontsize=7, color=INK)
        x += 0.34
end_frames = x - gap
# silent delay
x += 0.12
starts["silent"] = x
for k in range(3):
    ax.add_patch(Rectangle((x, y0), bs, bs, fc=SIL, ec=MUTED, lw=0.7, zorder=3))
    x += bs + gap
    if k == 1:
        ax.text(x + 0.12, y0 + bs / 2, r"$\cdots$", ha="center", va="center", fontsize=7, color=INK)
        x += 0.34
end_silent = x - gap
# go cue and response window
x += 0.12
starts["go"] = x
for k in range(3):
    ax.add_patch(Rectangle((x, y0), bs, bs, fc=ACC_FILL, ec=ACC, lw=0.8, zorder=3))
    ax.text(x + bs / 2, y0 + bs / 2, "go", ha="center", va="center", fontsize=7.0, color=INK, zorder=4)
    x += bs + gap
    if k == 1:
        ax.text(x + 0.12, y0 + bs / 2, r"$\cdots$", ha="center", va="center", fontsize=7, color=INK)
        x += 0.34
end_go = x - gap

brace(starts["frames"], end_frames, y0 - 0.06, False, "20 event frames", MUTED)
brace(starts["silent"], end_silent, y0 - 0.06, False, r"$d$ silent steps", MUTED)
brace(starts["go"], end_go, y0 + bs + 0.06, True, "name the digit", ACC)
ax.annotate("", xy=(end_go + 0.35, -0.55), xytext=(starts["frames"], -0.55),
            arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED))
ax.text(end_go + 0.45, -0.55, r"$t$", ha="left", va="center", fontsize=8, color=MUTED)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
if a.panel:
    fig.text(0.003, 0.995, a.panel, ha="left", va="top", fontsize=9, fontweight="bold")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}")
print("saved", a.out, "label of sample:", int(d["y_test"][a.sample]))
