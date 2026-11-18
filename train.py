"""Delayed MNIST-DVS classification.

A sample is T_in = 20 frames of DVS events (25 ms each), then `d` silent steps, then a response window of
`window` steps in which `n_go` go lines are on.  The class is read from the output spike counts in the
response window (rate decoding, as in [1] of README.md); ties are broken at random.

Fixed delay:     --delays 40
Variable delay:  --delays 0,10,20,40,80,160   (each training batch draws one delay; the network is not told
                 which, so it must hold the class until the go cue whenever it comes)
Model selection: best epoch by validation accuracy averaged over --eval_delays (default = --delays); the
test set is evaluated once per delay with that checkpoint (`test_reps` single-shot repetitions).

Usage: python train.py --model sqs_event --delays 0,10,20,40,80,160 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import Net, hidden_for_budget

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["sqs_event", "sqs_read", "sqs_chen", "lif", "lif_lb", "qlif"])
p.add_argument("--delays", default="0", help="comma-separated training delays (steps)")
p.add_argument("--eval_delays", default="", help="comma-separated evaluation delays (default: --delays)")
p.add_argument("--window", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--n_go", type=int, default=32, help="number of go lines, all on in the response window")
p.add_argument("--ref_hidden", type=int, default=256, help="parameter budget = SQS (3 angles) with this hidden size")
p.add_argument("--hidden", type=int, default=0, help="override the budget-matched hidden size")
p.add_argument("--loss", default="sum_q", choices=["sum_q", "sum_spk", "step_q"])
p.add_argument("--theta_init", default="memory", choices=["uniform", "small", "memory"])
p.add_argument("--write_max", type=float, default=0.25, help="memory init: write angle ~ U(0, write_max*pi)")
p.add_argument("--n_mem", type=int, default=1, help="memory qubits per SQS neuron (event ansatz)")
p.add_argument("--beta", type=float, default=0.95, help="LIF leak (1.0 = non-leaky integrate-and-fire)")
p.add_argument("--bias", type=int, default=1, help="LIF synapses with bias (1) or without (0)")
p.add_argument("--read_min", type=float, default=0.75, help="memory init: read angle ~ U(read_min*pi, pi)")
p.add_argument("--extra_test_delays", default="", help="delays evaluated on the test set only (not trained)")
p.add_argument("--gain0", type=float, default=0.0, help="learnable logit gain with this initial value (0 = off)")
p.add_argument("--surrogate", default="sigmoid", choices=["sigmoid", "ste"])
p.add_argument("--w_scale", type=float, default=1.0, help="multiplier on the default weight init")
p.add_argument("--val_n", type=int, default=1000)
p.add_argument("--go_init", type=float, default=1.0,
               help="go synapses ~ U(0, 2c/n_go): the go cue drives each hidden neuron with current ~c (0 = default init)")
p.add_argument("--no_test", action="store_true", help="skip the test evaluation (hyper-parameter search)")
p.add_argument("--test_reps", type=int, default=3, help="repetitions of the (stochastic) test evaluation")
p.add_argument("--n_train", type=int, default=0, help="subsample the training set (0 = all 8000)")
p.add_argument("--threads", type=int, default=2)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--impl", default="real", choices=["real", "complex"], help="SQS simulator (identical results)")
p.add_argument("--compile", type=int, default=1, help="torch.compile the real SQS step (GPU only)")
p.add_argument("--tag", default="main")
p.add_argument("--out", default=os.path.join(HERE, "results"))
a = p.parse_args()

train_delays = [int(v) for v in a.delays.split(",")]
eval_delays = [int(v) for v in a.eval_delays.split(",")] if a.eval_delays else train_delays
torch.set_num_threads(a.threads)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
rng = np.random.RandomState(a.seed)

# ---------------------------------------------------------------- data
d = np.load(os.path.join(HERE, "data", "mnist_dvs_T20.npz"))
n_ch, T_in = int(d["n_ch"]), int(d["T"])
x_all = np.unpackbits(d["x_train"], axis=-1)[..., :n_ch]
y_all = d["y_train"]
x_te = np.unpackbits(d["x_test"], axis=-1)[..., :n_ch]
y_te = d["y_test"]
perm = np.random.RandomState(1234).permutation(len(y_all))          # fixed split for every run
val_idx, tr_idx = perm[:a.val_n], perm[1000:]
if a.n_train:
    tr_idx = tr_idx[: a.n_train]
dev = torch.device(a.device)
x_tr, y_tr, x_va, y_va = x_all[tr_idx], y_all[tr_idx], x_all[val_idx], y_all[val_idx]
x_tr, x_va, x_te = (torch.from_numpy(v).to(dev) for v in (x_tr, x_va, x_te))   # uint8 on the device
y_tr, y_va, y_te = (torch.from_numpy(v).to(dev) for v in (y_tr, y_va, y_te))

n_in, n_out = n_ch + a.n_go, 10                                       # + go lines
budget = n_in * a.ref_hidden + a.ref_hidden * n_out + 3 * (a.ref_hidden + n_out)


def make_seq(xb, delay):
    """[B, T_in, n_ch] uint8 -> [T, B, n_in] float (frames, silent delay, go window) and the silent mask."""
    T = T_in + delay + a.window
    x = torch.zeros(T, xb.shape[0], n_in, device=xb.device)
    x[:T_in, :, :n_ch] = xb.float().transpose(0, 1)
    x[T_in + delay:, :, n_ch:] = 1.0
    return x, [T_in <= t < T_in + delay for t in range(T)]


# ---------------------------------------------------------------- model
hidden, _ = (a.hidden, None) if a.hidden else hidden_for_budget(a.model, n_in, n_out, budget, n_mem=a.n_mem)
kw = ({"theta_init": a.theta_init, "surrogate": a.surrogate, "impl": a.impl, "write_max": a.write_max * np.pi, "read_min": a.read_min * np.pi, "n_mem": a.n_mem,
       "compiled": bool(a.compile) and a.impl == "real" and a.device.startswith("cuda")}
      if a.model.startswith("sqs_") else ({"beta": a.beta, "bias": bool(a.bias)} if a.model.startswith("lif") else {}))
net = Net(a.model, n_in, hidden, n_out, **kw).to(dev)
if a.w_scale != 1.0:
    with torch.no_grad():
        for mod in (net.l1, net.l2):
            for nme, prm in mod.named_parameters():
                if "weight" in nme:
                    prm.mul_(a.w_scale)
if a.go_init > 0:   # after w_scale, so the go drive is the same for every weight scale
    with torch.no_grad():
        lin = net.l1.fcq if a.model == "qlif" else net.l1.fc
        lin.weight[:, n_ch:].uniform_(0.0, 2.0 * a.go_init / a.n_go)
gain = torch.nn.Parameter(torch.tensor(a.gain0 if a.gain0 > 0 else 1.0, device=dev), requires_grad=a.gain0 > 0)
opt = torch.optim.Adam(list(net.parameters()) + ([gain] if a.gain0 > 0 else []), lr=a.lr)
n_par = net.n_params() + int(a.gain0 > 0)


def loss_fn(spk, q, y):
    if a.loss == "sum_q":         # Born probability (SQS) / membrane (LIF, QLIF), summed over the window
        return F.cross_entropy(gain * q.sum(0), y)
    if a.loss == "sum_spk":       # spike count (surrogate gradient), summed over the window
        return F.cross_entropy(gain * spk.sum(0), y)
    return sum(F.cross_entropy(gain * q[t], y) for t in range(q.shape[0]))


@torch.no_grad()
def evaluate(x, y, delays, reps=1):
    """Spike-count accuracy (and readout-quantity accuracy) per delay, plus spikes per step."""
    net.eval()
    out = {}
    for dl in delays:
        c_spk = c_q = 0.0
        nspk, nsteps = 0.0, 0
        for _ in range(reps):
            for i in range(0, len(y), 250):
                xb, sil = make_seq(x[i:i + 250], dl)
                yb = y[i:i + 250]
                spk, q, n_spk = net(xb, a.window, sil)
                counts = spk.sum(0) + 0.5 * torch.rand_like(spk[0])    # random tie-break
                c_spk += (counts.argmax(-1) == yb).sum().item()
                c_q += (q.sum(0).argmax(-1) == yb).sum().item()
                nspk += n_spk.sum().item()
                nsteps += len(yb) * xb.shape[0]
        out[dl] = {"acc": c_spk / (reps * len(y)), "acc_q": c_q / (reps * len(y)), "spikes_per_step": nspk / nsteps}
    net.train()
    return out


# ---------------------------------------------------------------- train
name = f"{a.tag}_{a.model}{'_m%d' % a.n_mem if a.n_mem > 1 else ''}_d{'-'.join(map(str, train_delays))}_s{a.seed}"
os.makedirs(a.out, exist_ok=True)
hist = []
best = {"val": -1.0}
best_state = None
t0 = time.time()
print(f"{name}: hidden={hidden}, params={n_par}, train delays {train_delays}, eval delays {eval_delays}", flush=True)
for ep in range(1, a.epochs + 1):
    order = torch.from_numpy(rng.permutation(len(y_tr))).to(dev)
    tot, nb = 0.0, 0
    for i in range(0, len(order), a.batch):
        idx = order[i:i + a.batch]
        xb, sil = make_seq(x_tr[idx], train_delays[rng.randint(len(train_delays))])
        spk, q, _ = net(xb, a.window, sil)
        loss = loss_fn(spk, q, y_tr[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        tot += loss.detach()
        nb += 1
    va = evaluate(x_va, y_va, eval_delays)
    va_mean = float(np.mean([va[dl]["acc"] for dl in eval_delays]))
    rec = {"epoch": ep, "loss": float(tot) / nb, "val_mean": va_mean,
           "val": {str(k): v["acc"] for k, v in va.items()}, "gain": gain.item(), "time": time.time() - t0}
    hist.append(rec)
    if va_mean > best["val"]:
        best = {"val": va_mean, "epoch": ep}
        best_state = copy.deepcopy(net.state_dict()), gain.item()
    print(f"{name} ep{ep:3d} loss {rec['loss']:.4f} val {va_mean:.3f} | "
          + " ".join(f"d{k}:{v['acc']:.2f}" for k, v in va.items()) + f" | gain {gain.item():.2f} [{rec['time']:.0f}s]",
          flush=True)

res = {"name": name, "args": vars(a), "hidden": hidden, "params": n_par, "train_delays": train_delays,
       "eval_delays": eval_delays, "best": best, "history": hist}
if not a.no_test:
    test_delays = eval_delays + [int(v) for v in a.extra_test_delays.split(",") if v]
    final_test = evaluate(x_te, y_te, test_delays, a.test_reps)
    net.load_state_dict(best_state[0])
    with torch.no_grad():
        gain.fill_(best_state[1])
    best_test = evaluate(x_te, y_te, test_delays, a.test_reps)
    res["test"] = {str(k): v for k, v in best_test.items()}
    res["test_final_epoch"] = {str(k): v for k, v in final_test.items()}
    print(f"{name} TEST (best-val epoch {best['epoch']}): "
          + " ".join(f"d{k}:{v['acc']:.3f}" for k, v in best_test.items()), flush=True)
with open(os.path.join(a.out, name + ".json"), "w") as f:
    json.dump(res, f, indent=1)
print(f"{name} DONE", flush=True)
