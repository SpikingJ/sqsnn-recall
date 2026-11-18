"""Summarize runs of train.py.  Usage: python summarize.py <results_dir> [prefix]"""
import glob
import json
import os
import sys
from collections import defaultdict

rd = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
rows = defaultdict(list)
for f in sorted(glob.glob(os.path.join(rd, f"{prefix}*.json"))):
    r = json.load(open(f))
    h = r["history"]
    best_ep = r["best"]["epoch"]
    val_best = h[best_ep - 1]["val"]
    test = r.get("test")
    rows[r["args"]["model"]].append((r["best"]["val"], best_ep, len(h), val_best, test, r["args"]["tag"],
                                     r["train_delays"], r["args"]["seed"], h[-1]["time"], r["params"]))
for m, rs in rows.items():
    print(f"=== {m}")
    for v, ep, n, vb, test, tag, tds, s, t, npar in sorted(rs, key=lambda z: -z[0]):
        per = " ".join(f"d{k}:{x:.2f}" for k, x in vb.items())
        line = f"  val {v:.3f} @ep{ep:3d}/{n:3d} [{per}]  train d={tds} s={s} {tag} params={npar} [{t:.0f}s]"
        if test:
            line += "\n      TEST " + " ".join(f"d{k}:{x['acc']:.3f}" for k, x in test.items())
        print(line)
