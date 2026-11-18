"""Bin MNIST-DVS events into binary frames, as in the SQS paper [1, Sec. V-A3]
(references in README.md).

26 x 26 pixels, polarity merged (676 channels), dt = 25 ms, first T = 20 bins (0-500 ms).
A channel is 1 in a bin if the pixel emitted at least one event in that bin.
Input: mnist_dvs_events.hdf5, made from the MNIST-DVS recordings by neurodata/preprocessing/process_mnistdvs.py
of https://github.com/kclip/neurodata (see README.md).
Output: data/mnist_dvs_T20.npz with packed bits x_train [9000, 20, 676], y_train, x_test, y_test.
"""
import argparse
import os

import h5py
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist_dvs_events.hdf5"),
               help="MNIST-DVS events in HDF5 (see README.md)")
p.add_argument("--T", type=int, default=20)
p.add_argument("--dt_us", type=int, default=25000)
p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "mnist_dvs_T20.npz"))
a = p.parse_args()

f = h5py.File(a.src, "r")
side = int(f["stats/train_data"][1])          # 26
out = {}
for split in ("train", "test"):
    g = f[split]
    labels = g["labels"][()].astype(np.int64).ravel()
    n = len(labels)
    x = np.zeros((n, a.T, side * side), dtype=np.uint8)
    for i in range(n):
        ev = g[str(i)][()]
        t, xx, yy = ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64), ev[:, 2].astype(np.int64)
        b = t // a.dt_us
        keep = b < a.T
        x[i, b[keep], (xx[keep] * side + yy[keep])] = 1
    out[f"x_{split}"] = np.packbits(x, axis=-1)
    out[f"y_{split}"] = labels
    rate = x.mean()
    print(f"{split}: n={n}, frames {x.shape}, mean channel activity per bin {rate:.4f}, "
          f"active channels per bin {x.sum(-1).mean():.1f}, labels {np.bincount(labels)}")
np.savez_compressed(a.out, n_ch=side * side, T=a.T, **out)
print("saved", a.out, os.path.getsize(a.out) / 1e6, "MB")
