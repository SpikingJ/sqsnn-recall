"""Circuit panel: one time step of the SQS neuron with the proposed event-gated read and n memory qubits.

The write gates (controlled X rotations from the IO qubit, angles w_1..w_n) share their control and
commute, so they are drawn in one column; the read gates (Z rotations of the IO qubit controlled by
memory qubit n, angles r_n) are drawn one per memory qubit; m_n are the angles of the memory X rotations
(n = 1..N).
Every rotation box names its axis and carries its angle as subscript (X_theta, Z_theta).
Usage: python draw_circuit_nm.py --out figures/panel_c --panel "(c)" --width 2.95 --height 1.3
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

p = argparse.ArgumentParser()
p.add_argument("--panel", default="")
p.add_argument("--width", type=float, default=3.45)
p.add_argument("--height", type=float, default=1.75)
p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "circuit_event_nm"))
a = p.parse_args()

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix"})
INK, ACC, ACC_FILL = "#0b0b0b", "#eb6834", "#fbe1d6"
fig, ax = plt.subplots(figsize=(a.width, a.height), dpi=300)
ax.set_xlim(0, 11.4)
ax.set_ylim(-0.3, 3.55)
ax.axis("off")
y_io, y_m1, y_mn, bw, bh = 2.55, 1.45, 0.25, 0.95, 0.6


def box(x, y, text, fill="white", edge=INK):
    ax.add_patch(Rectangle((x - bw / 2, y - bh / 2), bw, bh, fc=fill, ec=edge, lw=0.9, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.0, zorder=4)


def dot(x, y, color=INK):
    ax.plot([x], [y], "o", ms=4.0, color=color, zorder=4)


x_end_io, x_end_m = 9.0, 10.75
ax.plot([1.45, x_end_io], [y_io, y_io], color=INK, lw=0.9, zorder=1)
for y in (y_m1, y_mn):
    ax.plot([1.45, x_end_m], [y, y], color=INK, lw=0.9, zorder=1)
ax.text(1.35, y_io, r"IO $|0\rangle$", ha="right", va="center", fontsize=8.0)
ax.text(1.35, y_m1, r"memory 1", ha="right", va="center", fontsize=8.0)
ax.text(1.35, y_mn, r"memory $N$", ha="right", va="center", fontsize=8.0)
ax.text(0.85, (y_m1 + y_mn) / 2 + 0.04, r"$\vdots$", ha="center", va="center", fontsize=9)
ax.text(x_end_m + 0.08, (y_m1 + y_mn) / 2, r"$\rho_t$", ha="left", va="center", fontsize=8.5)
ax.text(1.62, (y_m1 + y_mn) / 2, r"$\rho_{t-1}$", ha="left", va="center", fontsize=7.5, color="#52514e")

box(2.65, y_io, r"$X_{\phi_t/2}$")                                 # input half pulse
dot(3.85, y_io)                                                    # writes, one column
ax.plot([3.85, 3.85], [y_io, y_mn + bh / 2], color=INK, lw=0.9, zorder=2)
box(3.85, y_m1, r"$X_{w_1}$")
box(3.85, y_mn, r"$X_{w_N}$")
for x, y, lab in ((5.0, y_m1, r"$Z_{r_1}$"), (6.6, y_mn, r"$Z_{r_N}$")):    # reads, highlighted
    ax.plot([x, x], [y, y_io - bh / 2], color=ACC, lw=1.1, zorder=2)
    dot(x, y, ACC)
    box(x, y_io, lab, fill=ACC_FILL, edge=ACC)
# one read gate per memory qubit: the gates of memory qubits 2, ..., N-1 are left out between the two
ax.text(5.8, y_io, r"$\cdots$", ha="center", va="center", fontsize=9, color=ACC, zorder=4,
        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none"))
box(7.85, y_io, r"$X_{\phi_t/2}$")                                 # second input half pulse
box(7.85, y_m1, r"$X_{m_1}$")
box(7.85, y_mn, r"$X_{m_N}$")
mx = 9.15                                                          # measurement
ax.add_patch(Rectangle((mx - bw / 2, y_io - bh / 2), bw, bh, fc="white", ec=INK, lw=0.9, zorder=3))
# meter: a semicircle and a needle that clearly passes through it
ax.add_patch(Arc((mx, y_io - 0.2), 0.6, 0.48, theta1=0, theta2=180, lw=0.9, color=INK, zorder=4))
ax.annotate("", xy=(mx + 0.36, y_io + 0.26), xytext=(mx, y_io - 0.2),
            arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
ax.plot([mx + bw / 2, mx + bw / 2 + 0.3], [y_io, y_io], color=INK, lw=0.9, zorder=1)
ax.text(mx + bw / 2 + 0.4, y_io, r"spike $s_t$", ha="left", va="center", fontsize=8.0)
for x, t, c in [(2.65, "input", INK), (3.85, "write", INK), (5.8, "read", ACC), (7.85, "input", INK)]:
    ax.text(x, 3.3, t, ha="center", va="center", fontsize=7.5, color=c, fontweight="bold" if c == ACC else None)
fig.tight_layout(pad=0.05)
if a.panel:
    fig.text(0.003, 0.995, a.panel, ha="left", va="top", fontsize=9, fontweight="bold")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{a.out}.{ext}")
print("saved", a.out)
