"""Check the event ansatz with n_mem memory qubits (models.sqs_step_event_nm) against
(1) the verified n_mem = 1 step (sqs_step_real) and (2) an explicit numpy statevector simulation
with Kronecker-product gate matrices on 1 + n_mem qubits.  Usage: python check_nmem.py [cuda]"""
import sys

import numpy as np
import torch

from models import _get_step_nm, sqs_silent_real, sqs_step_real

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.manual_seed(0)
rng = np.random.default_rng(0)
B, K = 5, 4
I2 = np.eye(2)
P0, P1 = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])


def rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def kron_all(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def ref_step(m, phi, th, n):
    """Qubit 0 = IO, qubits 1..n = memory (big-endian). Returns P(spike), post-measurement memories."""
    nq = 1 + n
    single = lambda U, q: kron_all([U if k == q else I2 for k in range(nq)])
    ctrl = lambda U, c, t: kron_all([P0 if k == c else I2 for k in range(nq)]) + \
        kron_all([P1 if k == c else (U if k == t else I2) for k in range(nq)])
    psi = np.kron(np.array([1.0, 0.0]), m)
    psi = single(rx(phi / 2), 0) @ psi
    for q in range(n):
        psi = ctrl(rx(th[q]), 0, 1 + q) @ psi
    for q in range(n):
        psi = ctrl(rz(th[n + q]), 1 + q, 0) @ psi
    psi = single(rx(phi / 2), 0) @ psi
    for q in range(n):
        psi = single(rx(th[2 * n + q]), 1 + q) @ psi
    D = 1 << n
    a0, a1 = psi[:D], psi[D:]
    p1 = float(np.sum(np.abs(a1) ** 2))
    return p1, a0 / np.linalg.norm(a0), (a1 / np.linalg.norm(a1) if p1 > 1e-12 else None)


worst = 0.0
for compiled in ([False, True] if dev == "cuda" else [False]):
    # (1) n_mem = 1 against the verified step
    step, silent = _get_step_nm(compiled, 1)
    z = torch.randn(B, K, device=dev)
    th = torch.rand(K, 3, device=dev) * 3.0
    m = torch.randn(B, K, 2, 2, device=dev)
    m = m / m.pow(2).sum((-1, -2), keepdim=True).sqrt()
    u = torch.rand(B, K, device=dev)
    s_a, m_a, p_a = step(z, m, th, u, 1, "sigmoid")
    s_b, m_b, p_b = sqs_step_real(z, m.reshape(B, K, 4), th, u, "event", "sigmoid")
    e1 = max((p_a - p_b).abs().max().item(), (m_a.reshape(B, K, 4) - m_b).abs().max().item(),
             (s_a - s_b).abs().max().item(),
             (silent(m, th, 1).reshape(B, K, 4) - sqs_silent_real(m.reshape(B, K, 4), th)).abs().max().item())
    worst = max(worst, e1)
    print(f"compiled={compiled} n_mem=1 vs verified step: {e1:.2e}")
    # (2) n_mem = 2, 3 against the numpy statevector
    for n in (2, 3):
        step, silent = _get_step_nm(compiled, n)
        D = 1 << n
        z = torch.rand(B, K, device=dev)
        th = torch.rand(K, 3 * n, device=dev) * 3.0
        mc = rng.normal(size=(B, K, D)) + 1j * rng.normal(size=(B, K, D))
        mc /= np.linalg.norm(mc, axis=-1, keepdims=True)
        m = torch.tensor(np.stack((mc.real, mc.imag), -1), dtype=torch.float32, device=dev)
        u = torch.rand(B, K, device=dev)
        spk, new_m, p1 = step(z, m, th, u, n, "sigmoid")
        sil = silent(m, th, n)
        e = 0.0
        for b in range(B):
            for k in range(K):
                phi = np.pi * max(z[b, k].item(), 0.0)
                thk = th[k].cpu().numpy().astype(np.float64)
                p_ref, post0, post1 = ref_step(mc[b, k], phi, thk, n)
                e = max(e, abs(p_ref - p1[b, k].item()))
                post = post1 if spk[b, k].item() > 0.5 else post0
                got = new_m[b, k, :, 0].cpu().numpy() + 1j * new_m[b, k, :, 1].cpu().numpy()
                e = max(e, float(np.abs(got - post).max()))
                p_s, s0, _ = ref_step(mc[b, k], 0.0, thk, n)                 # silent step
                got_s = sil[b, k, :, 0].cpu().numpy() + 1j * sil[b, k, :, 1].cpu().numpy()
                e = max(e, p_s, float(np.abs(got_s - s0).max()))
        worst = max(worst, e)
        print(f"compiled={compiled} n_mem={n} vs numpy statevector: {e:.2e}")
print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
