# Training grid on Kaggle (free GPUs)

`kernel_train/train_kernel.py` is a Kaggle script kernel: it installs Ultralytics, copies the dataset once per GPU (the label
cache file is not safe to share between processes), splits the runs that are still missing across all visible GPUs, and writes one
JSON per run to `/kaggle/working/runs`. It skips finished runs, so a second version resumes from the first.

1. Build `kaggle/stage_data/synthmix-data.zip`: `real/` and `synth/` (images and labels), `splits.csv`, `synth_plan.csv`,
   `drift.csv`, `specs_all.json`, `specs_smoke.json` and `src/synthmix/runner.py` as `runner.py`.
2. `kaggle datasets create -p kaggle/stage_data` (private dataset `synthmix-data`).
3. `kaggle kernels push -p kaggle/kernel_train` (needs GPU T4 x2 and internet for the Ultralytics install).
4. `kaggle kernels output <you>/synthmix-train -p out` and copy `out/runs/*.json` to `results/runs/`.

Kaggle allows two GPU sessions at once and 12 hours per session; the grid (31 runs, 20 epochs) took about 4 hours for the first half and 3 hours for the rest.
Edit the ids in the two metadata files to use your own account; to resume a partial run, add your own earlier kernel to `kernel_sources`
in `kernel-metadata.json` and push again.
