# Quick physics check of three SQS ansaetze (one IO qubit, one memory qubit), statevector psi[a_io, m_mem].
import numpy as np
rng = np.random.default_rng(0)
def rx(t): c, s = np.cos(t/2), np.sin(t/2); return np.array([[c, -1j*s], [-1j*s, c]])
def rz(t): return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
def on_io(psi, U): return U @ psi            # psi shape (2_io, 2_mem)
def on_mem(psi, U): return psi @ U.T
def ctrl_io_on_mem(psi, U): psi = psi.copy(); psi[1, :] = U @ psi[1, :]; return psi
def ctrl_mem_on_io(psi, U): psi = psi.copy(); psi[:, 1] = U @ psi[:, 1]; return psi

def step(psi, phi, th, ansatz):
    if ansatz == "chen":    # original SQS ansatz [1, Fig. 6]
        psi = on_io(psi, rx(phi)); psi = ctrl_io_on_mem(psi, rx(th[0]))
        psi = on_io(psi, rx(th[1])); psi = on_mem(psi, rx(th[2]))
    elif ansatz == "read":  # abandoned paper: + CRX(mem -> IO)
        psi = on_io(psi, rx(phi)); psi = ctrl_io_on_mem(psi, rx(th[0]))
        psi = ctrl_mem_on_io(psi, rx(th[3])); psi = on_io(psi, rx(th[1])); psi = on_mem(psi, rx(th[2]))
    elif ansatz == "event": # proposed: split input pulse around write + controlled-phase read
        psi = on_io(psi, rx(phi/2)); psi = ctrl_io_on_mem(psi, rx(th[0]))
        psi = ctrl_mem_on_io(psi, rz(th[1])); psi = on_io(psi, rx(phi/2)); psi = on_mem(psi, rx(th[2]))
    return psi

def p_spike(psi): return float(np.sum(np.abs(psi[1, :])**2))
def collapse(psi, s):
    m = psi[s, :] / np.linalg.norm(psi[s, :]); out = np.zeros((2, 2), complex); out[0, :] = m; return out
def rand_mem():
    v = rng.normal(size=2) + 1j*rng.normal(size=2); v /= np.linalg.norm(v)
    out = np.zeros((2, 2), complex); out[0, :] = v; return out

for ansatz in ["chen", "read", "event"]:
    dp_in, dp_sil, backaction = 0.0, 0.0, 0.0
    for _ in range(300):
        th = rng.uniform(0, 2*np.pi, 4); phi = rng.uniform(0.2, np.pi)
        ps_in = [p_spike(step(rand_mem(), phi, th, ansatz)) for _ in range(20)]
        ps_sil = [p_spike(step(rand_mem(), 0.0, th, ansatz)) for _ in range(20)]
        dp_in = max(dp_in, max(ps_in) - min(ps_in)); dp_sil = max(dp_sil, max(ps_sil) - min(ps_sil))
        # back-action in silence: does the post-measurement memory Z-population depend on the outcome?
        psi0 = rand_mem(); pre = step(psi0, 0.0, th, ansatz)
        for s in (0, 1):
            if np.linalg.norm(pre[s, :]) > 1e-6:
                post = collapse(pre, s)
                ref = on_mem(psi0, rx(th[2]))  # pure unitary evolution of the memory (no measurement effect)
                if ansatz == "event":
                    ref = on_mem(ctrl_mem_on_io(psi0, rz(th[1])), rx(th[2]))
                backaction = max(backaction, abs(abs(post[0, 1])**2 - abs(ref[0, 1])**2))
    print(f"{ansatz:6s}  max memory-dependence of P(spike): with input {dp_in:.3f} | silent {dp_sil:.3f}"
          f" | silent back-action on mem population {backaction:.3f}")

# P(spike) of the event ansatz with phi=pi, write off, read angle pi, as a function of memory population
th = np.array([0.0, np.pi, 0.0, 0.0])
for p1 in [0, 0.25, 0.5, 0.75, 1]:
    psi = np.zeros((2, 2), complex); psi[0, 0] = np.sqrt(1-p1); psi[0, 1] = -1j*np.sqrt(p1)
    print(f"event, phi=pi, read=pi, mem P(1)={p1:.2f} -> P(spike)={p_spike(step(psi, np.pi, th, 'event')):.3f}")
