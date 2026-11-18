"""Spiking layers for the delayed MNIST-DVS memory task.

* SQSLayer  -- stochastic quantum spiking (SQS) neuron [1] with one input-output (IO) qubit, three ansaetze:
    "chen"  : original circuit [1, Fig. 6]: RX_io(phi) -> CRX(io->mem, t1) -> RX_io(t2), RX_mem(t3)
    "read"  : read-gate variant from an earlier draft of this work: chen + CRX(mem->io, t4) before the RX layer
    "event" : proposed event-gated read: RX_io(phi/2) -> CRX(io->mem, t1) -> CRZ(mem->io, t2)
              -> RX_io(phi/2), RX_mem(t3).  Without an input event (phi = 0) the IO qubit stays in |0>,
              so the memory is neither read nor disturbed; an input event reads it as a Ramsey sequence.
  Single-shot Born sampling of the IO qubit, collapse of the memory, sigmoid surrogate [1, App. A].
* LIFLayer  -- classical LIF neuron, same update as snnTorch Leaky (beta, subtract reset, fast sigmoid).
* QLIFLayer -- QLIF neuron [2], using the published AlphaSurrogate code verbatim (MIT license, see below).

References (the numbering of README.md):
[1] J. Chen, B. Rajendran, and O. Simeone, "Stochastic quantum spiking neural networks with quantum memory and
    local learning," IEEE J. Sel. Areas Commun., vol. 44, pp. 5421-5436, 2026, doi: 10.1109/JSAC.2026.3714333.
[2] D. Brand and F. Petruccione, "A quantum leaky integrate-and-fire spiking neuron and network," npj Quantum
    Inf., vol. 10, no. 1, Art. no. 125, Dec. 2024, doi: 10.1038/s41534-024-00921-x.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------------
# SQS
# --------------------------------------------------------------------------------------------
N_ANGLES = {"chen": 3, "read": 4, "event": 3}


def _rx_pair(c, ms, a0, a1):
    """RX on a two-level index: (a0, a1) -> (c a0 + ms a1, ms a0 + c a1), ms = -i sin."""
    return c * a0 + ms * a1, ms * a0 + c * a1


# ---- real-arithmetic version of the same step (complex numbers as (re, im) pairs), so that
# ---- torch.compile can fuse it; checked against the complex version in check_real_impl.py
def _rx_r(c, s, a0, a1):
    """(a0, a1) -> (c a0 - i s a1, -i s a0 + c a1) for complex pairs a = (re, im)."""
    (a0r, a0i), (a1r, a1i) = a0, a1
    return ((c * a0r + s * a1i, c * a0i - s * a1r),
            (s * a0i + c * a1r, -s * a0r + c * a1i))


def _phase_r(a, ang):
    """a * exp(i ang)."""
    ar, ai = a
    co, si = torch.cos(ang), torch.sin(ang)
    return (ar * co - ai * si, ar * si + ai * co)


def sqs_step_real(z, m, theta, u, ansatz: str, surrogate: str):
    """One SQS step.  z [B,K] input currents, m [B,K,4] memory (m0r, m0i, m1r, m1i),
    theta [K,G], u [B,K] uniform draws.  Returns spikes (surrogate attached), new memory, P(spike)."""
    phi = math.pi * torch.relu(z)
    half = phi / 4 if ansatz == "event" else phi / 2
    c, s = torch.cos(half), torch.sin(half)
    m0 = (m[..., 0], m[..., 1])
    m1 = (m[..., 2], m[..., 3])
    a00, a01 = (c * m0[0], c * m0[1]), (c * m1[0], c * m1[1])
    a10, a11 = (s * m0[1], -s * m0[0]), (s * m1[1], -s * m1[0])          # (-i s) m
    a10, a11 = _rx_r(torch.cos(theta[:, 0] / 2), torch.sin(theta[:, 0] / 2), a10, a11)   # write
    if ansatz == "event":
        a01, a11 = _phase_r(a01, -theta[:, 1] / 2), _phase_r(a11, theta[:, 1] / 2)      # CRZ read
        a00, a10 = _rx_r(c, s, a00, a10)                                               # 2nd half pulse
        a01, a11 = _rx_r(c, s, a01, a11)
    else:
        if ansatz == "read":
            a01, a11 = _rx_r(torch.cos(theta[:, 3] / 2), torch.sin(theta[:, 3] / 2), a01, a11)
        c2, s2 = torch.cos(theta[:, 1] / 2), torch.sin(theta[:, 1] / 2)
        a00, a10 = _rx_r(c2, s2, a00, a10)
        a01, a11 = _rx_r(c2, s2, a01, a11)
    c3, s3 = torch.cos(theta[:, 2] / 2), torch.sin(theta[:, 2] / 2)
    a00, a01 = _rx_r(c3, s3, a00, a01)
    a10, a11 = _rx_r(c3, s3, a10, a11)
    p1 = (a10[0] ** 2 + a10[1] ** 2 + a11[0] ** 2 + a11[1] ** 2).clamp(0.0, 1.0)
    s_hard = (u < p1.detach()).to(p1.dtype)
    soft = torch.sigmoid(p1) if surrogate == "sigmoid" else p1
    spk = s_hard + (soft - soft.detach())
    sel = s_hard > 0.5
    inv = torch.rsqrt(torch.where(sel, p1, 1.0 - p1).clamp_min(1e-10))
    new_m = torch.stack((torch.where(sel, a10[0], a00[0]) * inv, torch.where(sel, a10[1], a00[1]) * inv,
                         torch.where(sel, a11[0], a01[0]) * inv, torch.where(sel, a11[1], a01[1]) * inv),
                        dim=-1)
    return spk, new_m, p1


def sqs_silent_real(m, theta):
    """Event ansatz without input: memory <- RX(t3) diag(1, exp(-i t2/2)) memory, no spike."""
    m0 = (m[..., 0], m[..., 1])
    m1 = _phase_r((m[..., 2], m[..., 3]), -theta[:, 1] / 2)
    m0, m1 = _rx_r(torch.cos(theta[:, 2] / 2), torch.sin(theta[:, 2] / 2), m0, m1)
    return torch.stack((m0[0], m0[1], m1[0], m1[1]), dim=-1)


# ---- event ansatz with n_mem memory qubits (one IO qubit).  Memory register m [B, K, 2**n_mem, 2]
# ---- (last axis re, im), basis index j with memory qubit 0 as the most significant bit.
# ---- theta [K, 3 n_mem] = (write angles, read angles, memory RX angles), one per memory qubit:
# ----   RX_io(phi/2) -> CRX(io -> mem q, w_q) for all q -> CRZ(mem q -> io, r_q) for all q
# ----   -> RX_io(phi/2) -> RX(mem q, m_q) for all q.   For n_mem = 1 this is the "event" ansatz above.
def _rx_qubit(ar, ai, angle, q, n_mem):
    """RX(angle) on memory qubit q of amplitudes [B, K, 2**n_mem] (re, im); angle [K]."""
    B, K, D = ar.shape
    shp = (B, K, 1 << q, 2, D >> (q + 1))
    ar, ai = ar.reshape(shp), ai.reshape(shp)
    c = torch.cos(angle / 2).reshape(1, K, 1, 1)
    s = torch.sin(angle / 2).reshape(1, K, 1, 1)
    (x0r, x0i), (x1r, x1i) = _rx_r(c, s, (ar[:, :, :, 0], ai[:, :, :, 0]), (ar[:, :, :, 1], ai[:, :, :, 1]))
    return (torch.stack((x0r, x1r), dim=3).reshape(B, K, D),
            torch.stack((x0i, x1i), dim=3).reshape(B, K, D))


def _mem_bits(n_mem, device):
    j = torch.arange(1 << n_mem, device=device)
    return ((j[:, None] >> torch.arange(n_mem - 1, -1, -1, device=device)[None, :]) & 1).float()  # [D, n]


def sqs_step_event_nm(z, m, theta, u, n_mem: int, surrogate: str):
    phi = math.pi * torch.relu(z)
    c, s = torch.cos(phi / 4)[..., None], torch.sin(phi / 4)[..., None]
    mr, mi = m[..., 0], m[..., 1]
    a0r, a0i = c * mr, c * mi
    a1r, a1i = s * mi, -s * mr                                            # (-i s) m
    for q in range(n_mem):                                                # writes, IO = 1 branch
        a1r, a1i = _rx_qubit(a1r, a1i, theta[:, q], q, n_mem)
    big = theta[:, n_mem:2 * n_mem] @ _mem_bits(n_mem, z.device).T       # [K, D] total read phase
    a0r, a0i = _phase_r((a0r, a0i), -big / 2)                             # CRZ reads (commute)
    a1r, a1i = _phase_r((a1r, a1i), big / 2)
    (a0r, a0i), (a1r, a1i) = _rx_r(c, s, (a0r, a0i), (a1r, a1i))         # 2nd half pulse
    for q in range(n_mem):                                                # memory rotations
        a0r, a0i = _rx_qubit(a0r, a0i, theta[:, 2 * n_mem + q], q, n_mem)
        a1r, a1i = _rx_qubit(a1r, a1i, theta[:, 2 * n_mem + q], q, n_mem)
    p1 = (a1r ** 2 + a1i ** 2).sum(-1).clamp(0.0, 1.0)
    s_hard = (u < p1.detach()).to(p1.dtype)
    soft = torch.sigmoid(p1) if surrogate == "sigmoid" else p1
    spk = s_hard + (soft - soft.detach())
    sel = (s_hard > 0.5)[..., None]
    inv = torch.rsqrt(torch.where(s_hard > 0.5, p1, 1.0 - p1).clamp_min(1e-10))[..., None]
    new_m = torch.stack((torch.where(sel, a1r, a0r) * inv, torch.where(sel, a1i, a0i) * inv), dim=-1)
    return spk, new_m, p1


def sqs_silent_event_nm(m, theta, n_mem: int):
    """No input event: phases of the read gates, then the memory rotations. No spike, no back-action."""
    big = theta[:, n_mem:2 * n_mem] @ _mem_bits(n_mem, m.device).T
    ar, ai = _phase_r((m[..., 0], m[..., 1]), -big / 2)
    for q in range(n_mem):
        ar, ai = _rx_qubit(ar, ai, theta[:, 2 * n_mem + q], q, n_mem)
    return torch.stack((ar, ai), dim=-1)


_COMPILED = {}


def _compiled_nm(n):
    """torch.compile closures with the number of memory qubits as a compile-time constant."""
    if ("nm", n) not in _COMPILED:
        def step(z, m, theta, u, surrogate):
            return sqs_step_event_nm(z, m, theta, u, n, surrogate)

        def silent(m, theta):
            return sqs_silent_event_nm(m, theta, n)

        _COMPILED[("nm", n)] = (torch.compile(step, dynamic=False), torch.compile(silent, dynamic=False))
    cs, cl = _COMPILED[("nm", n)]
    return (lambda z, m, theta, u, n_mem, surrogate: cs(z, m, theta, u, surrogate),
            lambda m, theta, n_mem: cl(m, theta))


def _get_step_nm(compiled, n_mem):
    """(step, silent) of the general event ansatz, signatures (z, m, theta, u, n_mem, surrogate) and
    (m, theta, n_mem)."""
    if not compiled:
        return sqs_step_event_nm, sqs_silent_event_nm
    return _compiled_nm(n_mem)


def _get_step(compiled):
    if not compiled:
        return sqs_step_real, sqs_silent_real
    if "step" not in _COMPILED:
        _COMPILED["step"] = torch.compile(sqs_step_real, dynamic=True)
        _COMPILED["silent"] = torch.compile(sqs_silent_real, dynamic=True)
    return _COMPILED["step"], _COMPILED["silent"]


class SQSLayer(nn.Module):
    def __init__(self, in_features, out_features, ansatz="event", theta_init="uniform",
                 surrogate="sigmoid", impl="complex", compiled=False, write_max=math.pi / 4,
                 read_min=0.75 * math.pi, n_mem=1, force_nm=False):
        super().__init__()
        assert ansatz in N_ANGLES
        self.in_features, self.out_features, self.ansatz = in_features, out_features, ansatz
        self.surrogate = surrogate
        self.impl, self.compiled = impl, compiled
        # general n_mem code path (event ansatz only); force_nm=True runs it also for n_mem = 1
        self.n_mem, self.nm = n_mem, (n_mem > 1 or force_nm)
        if self.nm:
            assert ansatz == "event" and impl == "real", "n_mem > 1 is implemented for the event ansatz"
        self.fc = nn.Linear(in_features, out_features, bias=False)  # z_t = w x_t  (paper Eq. 11)
        G = 3 * n_mem if self.nm else N_ANGLES[ansatz]
        self.theta = nn.Parameter(torch.empty(out_features, G))
        with torch.no_grad():
            if self.nm and theta_init == "memory":
                self.theta.normal_(0.0, 0.05)                                     # memory rotations
                self.theta[:, :n_mem].uniform_(0.0, write_max)                    # writes
                self.theta[:, n_mem:2 * n_mem].uniform_(read_min, math.pi)        # reads
            elif theta_init == "uniform":
                self.theta.uniform_(0.0, math.pi)
            elif theta_init == "memory":
                # write gently, read strongly, keep the memory rotation near the identity
                self.theta.normal_(0.0, 0.05)
                self.theta[:, 0].uniform_(0.0, write_max)                   # write CRX
                if ansatz == "event":
                    self.theta[:, 1].uniform_(read_min, math.pi)             # read CRZ
                elif ansatz == "read":
                    self.theta[:, 3].uniform_(read_min, math.pi)             # read CRX
            else:
                self.theta.normal_(0.0, 0.1)

    def init_state(self, B, device):
        # memory register only: the IO qubit is |0> at the start of every step
        if self.nm:
            m = torch.zeros(B, self.out_features, 1 << self.n_mem, 2, device=device)
            m[..., 0, 0] = 1.0
            return m
        if self.impl == "real":
            m = torch.zeros(B, self.out_features, 4, device=device)
        else:
            m = torch.zeros(B, self.out_features, 2, dtype=torch.complex64, device=device)
        m[..., 0] = 1.0
        return m

    def silent_step(self, mem):
        if self.nm:
            _, silent = _get_step_nm(self.compiled, self.n_mem)
            zeros = torch.zeros(mem.shape[:2], device=mem.device)
            return zeros, silent(mem, self.theta, self.n_mem), zeros
        if self.impl == "real":
            _, silent = _get_step(self.compiled)
            zeros = torch.zeros(mem.shape[:-1], device=mem.device)
            return zeros, silent(mem, self.theta), zeros
        return self._silent_step_complex(mem)

    def _silent_step_complex(self, mem):
        """Exact update of the event ansatz when the layer receives no input event (phi = 0):
        the IO qubit stays in |0>, so there is no spike and no back-action, and the memory only
        undergoes the fixed unitary RX(t3) diag(1, exp(-i t2/2)).  Gradients are identical to
        running `forward` with x = 0 (the input current and the spike probability are then 0)."""
        th = self.theta
        cplx = torch.complex64
        e = torch.exp(-0.5j * th[:, 1].to(cplx))
        m0, m1 = mem[..., 0], mem[..., 1] * e
        c3 = torch.cos(th[:, 2] / 2).to(cplx)
        s3 = (-1j) * torch.sin(th[:, 2] / 2).to(cplx)
        new_mem = torch.stack((c3 * m0 + s3 * m1, s3 * m0 + c3 * m1), dim=-1)
        zeros = torch.zeros(mem.shape[:-1], device=mem.device)
        return zeros, new_mem, zeros

    def forward(self, x, mem):
        if self.nm:
            z = self.fc(x)
            step, _ = _get_step_nm(self.compiled, self.n_mem)
            return step(z, mem, self.theta, torch.rand_like(z), self.n_mem, self.surrogate)
        if self.impl == "real":
            z = self.fc(x)
            step, _ = _get_step(self.compiled)
            return step(z, mem, self.theta, torch.rand_like(z), self.ansatz, self.surrogate)
        return self._forward_complex(x, mem)

    def _forward_complex(self, x, mem, u=None):
        z = self.fc(x)                                   # [B, K]
        phi = math.pi * torch.relu(z)                    # paper Eq. (12)
        th = self.theta
        m0, m1 = mem[..., 0], mem[..., 1]
        cplx = torch.complex64

        def cs(angle):  # cos(a/2), -i sin(a/2) as complex
            return torch.cos(angle / 2).to(cplx), (-1j) * torch.sin(angle / 2).to(cplx)

        if self.ansatz == "event":
            c, ms = cs(phi / 2)
        else:
            c, ms = cs(phi)
        # amplitudes a{io}{mem} after the input rotation on |0>_io (x) |m>
        a00, a01, a10, a11 = c * m0, c * m1, ms * m0, ms * m1
        # write: CRX(io -> mem, t1)
        c1, s1 = cs(th[:, 0])
        a10, a11 = _rx_pair(c1, s1, a10, a11)
        if self.ansatz == "chen":
            c2, s2 = cs(th[:, 1])
            (a00, a10), (a01, a11) = _rx_pair(c2, s2, a00, a10), _rx_pair(c2, s2, a01, a11)
        elif self.ansatz == "read":
            c4, s4 = cs(th[:, 3])                        # read gate CRX(mem -> io, t4)
            a01, a11 = _rx_pair(c4, s4, a01, a11)
            c2, s2 = cs(th[:, 1])
            (a00, a10), (a01, a11) = _rx_pair(c2, s2, a00, a10), _rx_pair(c2, s2, a01, a11)
        else:  # event: CRZ(mem -> io, t2), then the second half of the input pulse
            e = torch.exp(-0.5j * th[:, 1].to(cplx))
            a01, a11 = a01 * e, a11 * torch.conj(e)
            (a00, a10), (a01, a11) = _rx_pair(c, ms, a00, a10), _rx_pair(c, ms, a01, a11)
        c3, s3 = cs(th[:, 2])                            # RX_mem(t3)
        (a00, a01), (a10, a11) = _rx_pair(c3, s3, a00, a01), _rx_pair(c3, s3, a10, a11)

        def sq(a):
            return a.real * a.real + a.imag * a.imag

        p1 = (sq(a10) + sq(a11)).clamp(0.0, 1.0)         # Born P(spike), [B, K]
        with torch.no_grad():
            s_hard = ((torch.rand_like(p1) if u is None else u) < p1).to(p1.dtype)
        if self.surrogate == "sigmoid":                  # [1, App. A]
            soft = torch.sigmoid(p1)
        else:                                            # plain straight-through
            soft = p1
        spk = s_hard + (soft - soft.detach())
        # collapse onto the observed outcome, IO reset to |0>
        sel = s_hard.bool()
        n0 = torch.where(sel, a10, a00)
        n1 = torch.where(sel, a11, a01)
        p_sel = torch.where(sel, p1, 1.0 - p1).clamp_min(1e-10)
        norm = torch.sqrt(p_sel).to(cplx)
        new_mem = torch.stack((n0 / norm, n1 / norm), dim=-1)
        return spk, new_mem, p1


# --------------------------------------------------------------------------------------------
# LIF (snnTorch Leaky semantics)
# --------------------------------------------------------------------------------------------
class _FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, slope):
        ctx.save_for_backward(u)
        ctx.slope = slope
        return (u > 0).to(u.dtype)

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g / (ctx.slope * u.abs() + 1.0) ** 2, None


class LIFLayer(nn.Module):
    def __init__(self, in_features, out_features, beta=0.95, threshold=1.0, slope=25.0, bias=True,
                 learn_beta=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.beta, self.thr, self.slope = beta, threshold, slope
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.learn_beta = learn_beta
        if learn_beta:   # per-neuron leak, beta = sigmoid(b), initialised at `beta`
            self.beta_logit = nn.Parameter(torch.full((out_features,), math.log(beta / (1 - beta))))

    def init_state(self, B, device):
        return torch.zeros(B, self.out_features, device=device)

    def forward(self, x, mem):
        beta = torch.sigmoid(self.beta_logit) if self.learn_beta else self.beta
        reset = (mem > self.thr).to(mem.dtype).detach()          # snnTorch: reset from previous mem
        mem = beta * mem + self.fc(x) - reset * self.thr           # subtract reset
        spk = _FastSigmoidSpike.apply(mem - self.thr, self.slope)
        return spk, mem, mem


# --------------------------------------------------------------------------------------------
# QLIF (published code, https://github.com/deanbrand/QSNN/blob/main/alpha.py), verbatim
#
# The class AlphaSurrogate below is copied verbatim from that file. QLIFLayer wraps it in the same way
# as the Alpha module of that file, with the same threshold and T1. License of that repository:
#
# MIT License
#
# Copyright (c) 2024 Dean Brand
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# --------------------------------------------------------------------------------------------
class AlphaSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, theta, tau, x, mem, threshold, T1):
        ctx.save_for_backward(theta, tau, x, mem)
        ctx.threshold = threshold
        ctx.T1 = T1

        gamma = -2*torch.arcsin(torch.sqrt(mem * torch.exp(-tau/T1)))
        phi = torch.where(mem < threshold,
                          2 * torch.arcsin(torch.sqrt(mem)), 0.0)

        Theta = torch.sin((theta + phi) * x/2)**2
        Gamma = torch.sin((gamma + phi) * (1 - x)/2)**2
        mem = torch.where(mem < threshold, Gamma + Theta, 0.0)

        spike_out = torch.where(mem > threshold,
                                1.0, 0.0)

        mem = torch.where(mem > 1, 0.0, mem)

        return spike_out, mem

    @staticmethod
    def backward(ctx, grad_spike_out, grad_output):
        theta, tau, x, mem = ctx.saved_tensors
        T1 = ctx.T1
        grad_mem = 1 / (1 + (torch.pi * mem).pow_(2)) * grad_output

        sqrt_mem = torch.sqrt(mem)
        sqrt_mem_exp = torch.sqrt(mem * torch.exp(-tau/T1))

        dtheta = (x * torch.sin(x * (theta + 2 * torch.arcsin(sqrt_mem)))) / 2
        dtau = ((x - 1) * sqrt_mem_exp * torch.sin(2 * (x - 1) * (torch.arcsin(sqrt_mem) -
                torch.arcsin(sqrt_mem_exp)))) / (2 * T1 * torch.sqrt(1 - mem * torch.exp(-tau/T1)))

        grad_theta = (dtheta * grad_mem)
        grad_tau = (dtau * grad_mem)

        return grad_theta, grad_tau, grad_mem, None, None, None


class QLIFLayer(nn.Module):
    def __init__(self, in_features, out_features, threshold=0.75, T1=60.0, tau_scale=200.0):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.threshold, self.T1, self.tau_scale = threshold, T1, tau_scale
        self.fcq = nn.Linear(in_features, out_features)
        self.fct = nn.Linear(in_features, out_features)

    def init_state(self, B, device):
        return torch.zeros(B, self.out_features, device=device)

    def forward(self, x, mem):
        lin = self.fcq(x)
        theta = torch.abs(lin) * torch.pi
        tau = torch.abs(self.fct(x)) * self.tau_scale
        x_gate = torch.where(lin > 0, 1.0, 0.0)
        spk, mem = AlphaSurrogate.apply(theta, tau, x_gate, mem, self.threshold, self.T1)
        return spk, mem, mem


# --------------------------------------------------------------------------------------------
# two-layer feed-forward network
# --------------------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, model, n_in, n_hidden, n_out, **kw):
        super().__init__()
        self.model = model
        if model.startswith("sqs_"):
            ansatz = model[4:]
            self.l1 = SQSLayer(n_in, n_hidden, ansatz=ansatz, **kw)
            self.l2 = SQSLayer(n_hidden, n_out, ansatz=ansatz, **kw)
        elif model in ("lif", "lif_lb"):
            if model == "lif_lb":
                kw = dict(kw, learn_beta=True)
            self.l1 = LIFLayer(n_in, n_hidden, **kw)
            self.l2 = LIFLayer(n_hidden, n_out, **kw)
        elif model == "qlif":
            self.l1 = QLIFLayer(n_in, n_hidden, **kw)
            self.l2 = QLIFLayer(n_hidden, n_out, **kw)
        else:
            raise ValueError(model)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, window, silent=None):
        """x: [T, B, n_in]. Returns output spikes, readout quantity (p / membrane) over the last
        `window` steps ([window, B, n_out] each), and the total hidden+output spike count per sample.
        `silent[t]` marks steps whose input is all zero (used by the exact fast path of the event
        ansatz; other models simulate every step)."""
        T, B, _ = x.shape
        s1, s2 = self.l1.init_state(B, x.device), self.l2.init_state(B, x.device)
        spk_out, q_out = [], []
        n_spk = torch.zeros(B, device=x.device)
        fast = self.model == "sqs_event"
        for t in range(T):
            if fast and silent is not None and silent[t]:
                # no input event: the hidden layer emits no spike, so the output layer is silent too
                h, s1, _ = self.l1.silent_step(s1)
                o, s2, q = self.l2.silent_step(s2)
            else:
                h, s1, _ = self.l1(x[t], s1)
                o, s2, q = self.l2(h, s2)
            n_spk = n_spk + h.detach().sum(-1) + o.detach().sum(-1)
            if t >= T - window:
                spk_out.append(o)
                q_out.append(q)
        return torch.stack(spk_out), torch.stack(q_out), n_spk


def hidden_for_budget(model, n_in, n_out, budget, n_mem=1):
    """Hidden size whose parameter count is closest to `budget`."""
    best = None
    for h in range(8, 2048):
        if model.startswith("sqs_"):
            g = 3 * n_mem if (model == "sqs_event" and n_mem > 1) else N_ANGLES[model[4:]]
            n = n_in * h + h * n_out + g * (h + n_out)
        elif model == "lif":
            n = (n_in + 1) * h + (h + 1) * n_out          # with bias (default)
        elif model == "lif_lb":
            n = (n_in + 2) * h + (h + 2) * n_out
        else:  # qlif: two weight planes with bias
            n = 2 * ((n_in + 1) * h + (h + 1) * n_out)
        if best is None or abs(n - budget) < abs(best[1] - budget):
            best = (h, n)
    return best
