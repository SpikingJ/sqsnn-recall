"""Check that the real-arithmetic (and torch.compile'd) SQS step equals the complex reference,
for outputs and gradients, with shared uniform draws.  Usage: python check_real_impl.py [cuda]"""
import sys

import torch

from models import SQSLayer, sqs_silent_real, sqs_step_real, _get_step

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
torch.manual_seed(0)
B, K, P = 7, 5, 11


def c2r(m):
    return torch.stack((m[..., 0].real, m[..., 0].imag, m[..., 1].real, m[..., 1].imag), -1)


worst = 0.0
for compiled in ([False, True] if dev == "cuda" else [False]):
    step = _get_step(compiled)[0] if compiled else sqs_step_real
    silent = _get_step(compiled)[1] if compiled else sqs_silent_real
    for ansatz in ("event", "read", "chen"):
        L = SQSLayer(P, K, ansatz=ansatz, theta_init="uniform").to(dev)
        x = (torch.rand(B, P, device=dev) < 0.5).float()
        m = torch.randn(B, K, 2, dtype=torch.complex64, device=dev)
        m = m / m.abs().pow(2).sum(-1, keepdim=True).sqrt()
        u = torch.rand(B, K, device=dev)
        w = torch.randn(B, K, 4, device=dev)
        outs = []
        for impl in ("complex", "real"):
            L.zero_grad()
            if impl == "complex":
                spk, nm, p1 = L._forward_complex(x, m, u)
                nm = c2r(nm)
            else:
                spk, nm, p1 = step(L.fc(x), c2r(m), L.theta, u, ansatz, "sigmoid")
            loss = (nm * w).sum() + (p1 * w[..., 0]).sum() + (spk * w[..., 1]).sum()
            loss.backward()
            outs.append([spk.detach(), nm.detach(), p1.detach(), L.theta.grad.clone(), L.fc.weight.grad.clone()])
        err = max((o0 - o1).abs().max().item() for o0, o1 in zip(*outs))
        # silent step
        th = L.theta.detach()
        ref = c2r(L._silent_step_complex(m)[1])
        err_s = (silent(c2r(m), th) - ref).abs().max().item() if ansatz == "event" else 0.0
        worst = max(worst, err, err_s)
        print(f"compiled={compiled} {ansatz:5s}: max |real - complex| over outputs+grads = {err:.2e}"
              + (f", silent step {err_s:.2e}" if ansatz == "event" else ""))
print("PASS" if worst < 1e-4 else "FAIL", f"(worst {worst:.2e})")
