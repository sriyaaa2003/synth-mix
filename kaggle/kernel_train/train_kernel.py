"""Kaggle script kernel: the synth-mix training grid on every GPU it can see (resumes from earlier output)."""
import glob
import os
import shutil
import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.40"])
import torch  # noqa: E402
from ultralytics import settings  # noqa: E402

# Kaggle ships a Ray version that the Ultralytics Ray Tune callback cannot talk to; switch the optional integrations off
settings.update({"raytune": False, "wandb": False, "mlflow": False, "comet": False, "clearml": False, "tensorboard": False, "neptune": False})

import traceback  # noqa: E402

os.makedirs("/kaggle/working/logs", exist_ok=True)
try:
    src = os.path.dirname(glob.glob("/kaggle/input/**/specs_all.json", recursive=True)[0])
    print("data at", src, flush=True)
    data, work, out = "/kaggle/temp/data", "/kaggle/temp/work", "/kaggle/working/runs"
    os.makedirs("/kaggle/temp", exist_ok=True)
    os.makedirs(out, exist_ok=True)
    os.makedirs("/kaggle/working/logs", exist_ok=True)
    for f in glob.glob("/kaggle/input/**/runs/*.json", recursive=True):       # resume from an earlier version's output
        shutil.copy(f, out)
    print(len(os.listdir(out)), "runs already done;", torch.cuda.device_count(), "GPU(s)", flush=True)

    import json  # noqa: E402

    n = max(1, torch.cuda.device_count())
    allspecs = json.load(open(f"{src}/specs_all.json"))
    pending = [x for x in allspecs if not os.path.exists(f"{out}/{x['name']}.json")]   # split only what is left over the GPUs
    json.dump(pending, open("/kaggle/temp/specs_pending.json", "w"))
    print(len(pending), "runs pending:", [x["name"] for x in pending], flush=True)
    for g in range(n):                                  # one data copy per process: the label cache file is not safe to share
        shutil.copytree(src, f"{data}{g}", dirs_exist_ok=True)
    procs = []
    for g in range(n):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        lf = open(f"/kaggle/working/logs/grid_gpu{g}.log", "w")
        procs.append(subprocess.Popen([sys.executable, f"{data}{g}/runner.py", "--data", f"{data}{g}", "--specs", "/kaggle/temp/specs_pending.json",
                                       "--out", out, "--work", f"{work}{g}", "--part", str(g), "--of", str(n),
                                       "--workers", "2", "--deadline-hours", "10"], env=env, stdout=lf, stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    for f in glob.glob("/kaggle/working/*.pt"):
        os.remove(f)
    print("finished:", len(os.listdir(out)), "run files", flush=True)
except Exception:
    open("/kaggle/working/logs/error.txt", "w").write(traceback.format_exc())
    raise
