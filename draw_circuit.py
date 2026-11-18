"""Circuit panel: one time step of the SQS neuron with the proposed event-gated read.

IO |0> --[RX(phi/2)]--*--[RZ(theta2)]--[RX(phi/2)]--[measure]-- spike s_t
                      |       |
mem rho_{t-1} ---[RX(theta1)]--*-------------[RX(theta3)]------ rho_t
Usage: python draw_circuit.py --out figures/circuit_event
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc

p = argparse.ArgumentParser()
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "circuit_event"))
a = p.parse_args()

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
INK, ACC, ACC_FILL = "#0b0b0b", "#eb6834", "#fbe1d6"
fig, ax = plt.subplots(figsize=(3.45, 1.25), dpi=300)
ax.set_xlim(0, 10.6)
ax.set_ylim(-0.25, 2.55)
ax.axis("off")
y_io, y_m, bw, bh = 1.55, 0.45, 0.95, 0.62


def box(x, y, text, fill="white", edge=INK, bold=False):
    ax.add_patch(Rectangle((x - bw / 2, y - bh / 2), bw, bh, fc=fill, ec=edge, lw=0.9, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, zorder=4, fontweight="bold" if bold else None)


def dot(x, y, color=INK):
    ax.plot([x], [y], "o", ms=4.2, color=color, zorder=4)


# wires
ax.plot([1.25, 7.55 + 0.95 / 2 + 0.35], [y_io, y_io], color=INK, lw=0.9, zorder=1)
ax.plot([1.25, 10.0], [y_m, y_m], color=INK, lw=0.9, zorder=1)
ax.text(1.15, y_io, r"IO $|0\rangle$", ha="right", va="center", fontsize=8.5)
ax.text(1.15, y_m, r"memory $\rho_{t-1}$", ha="right", va="center", fontsize=8.5)
ax.text(10.05, y_m, r"$\rho_t$", ha="left", va="center", fontsize=8.5)

# input half pulse
box(2.05, y_io, r"$\phi_t/2$")
# write: control IO, target memory (X rotation)
dot(3.35, y_io)
ax.plot([3.35, 3.35], [y_io, y_m + bh / 2], color=INK, lw=0.9, zorder=2)
box(3.35, y_m, r"$\theta_1$")
# read: control memory, target IO (Z rotation), highlighted
ax.plot([4.75, 4.75], [y_m, y_io - bh / 2], color=ACC, lw=1.1, zorder=2)
dot(4.75, y_m, ACC)
box(4.75, y_io, r"$Z_{\theta_2}$", fill=ACC_FILL, edge=ACC)
# second input half pulse
box(6.15, y_io, r"$\phi_t/2$")
# memory rotation
box(6.15, y_m, r"$\theta_3$")
# measurement
mx = 7.55
ax.add_patch(Rectangle((mx - bw / 2, y_io - bh / 2), bw, bh, fc="white", ec=INK, lw=0.9, zorder=3))
ax.add_patch(Arc((mx, y_io - 0.2), 0.62, 0.5, theta1=0, theta2=180, lw=0.9, color=INK, zorder=4))
ax.annotate("", xy=(mx + 0.36, y_io + 0.27), xytext=(mx, y_io - 0.2),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
ax.text(8.5, y_io, r"spike $s_t$", ha="left", va="center", fontsize=8.5)

# stage labels
for x, t, c in [(2.05, "input", INK), (3.35, "write", INK), (4.75, "read", ACC), (6.15, "input", INK)]:
    ax.text(x, 2.25, t, ha="center", va="center", fontsize=8, color=c, fontweight="bold" if c == ACC else None)
fig.tight_layout(pad=0.05)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}")
print("saved", a.out)
