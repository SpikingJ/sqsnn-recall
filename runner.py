"""Run a list of train.py jobs in parallel on several GPUs.  Each line of the job file is the
argument string of one train.py call; a job is skipped if its log already ends with DONE.

Usage: python runner.py jobs.txt --logdir logs --gpus 0,1,2,3 --per_gpu 4
"""
import argparse
import os
import queue
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("jobs")
p.add_argument("--logdir", required=True)
p.add_argument("--gpus", default="0")
p.add_argument("--per_gpu", type=int, default=4)
a = p.parse_args()
os.makedirs(a.logdir, exist_ok=True)
jobs = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
slots = queue.Queue()
for k in range(a.per_gpu):
    for g in a.gpus.split(","):
        slots.put(g)


def log_name(args):
    return os.path.join(a.logdir, "_".join(x.lstrip("-") for x in shlex.split(args)
                                           if not x.startswith("/"))[:180] + ".log")


def run(args):
    log = log_name(args)
    if os.path.exists(log) and "DONE" in open(log).read():
        return f"skip {args}"
    gpu = slots.get()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        t0 = time.time()
        with open(log, "w") as f:
            r = subprocess.run([sys.executable, os.path.join(HERE, "train.py")] + shlex.split(args),
                               stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env=env)
        return f"[{r.returncode}] gpu{gpu} {time.time() - t0:6.0f}s {args}"
    finally:
        slots.put(gpu)


with ThreadPoolExecutor(slots.qsize()) as ex:
    for msg in ex.map(run, jobs):
        print(msg, flush=True)
print("ALL DONE", flush=True)
