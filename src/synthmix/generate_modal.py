"""Weather/night edits of real BDD100K training images on a Modal GPU (Stable Diffusion 1.5 + depth and canny ControlNets).

    modal run src/synthmix/generate_modal.py --limit 6        # smoke test
    modal run src/synthmix/generate_modal.py                  # all n_source images

The depth + canny combination is the setting that kept geometry best in my earlier drift study
(driving-geometry): the labels of the source image are reused for the edited image, so geometry must not move.
"""
from __future__ import annotations

import io
import tomllib
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/")
CFG = tomllib.loads((ROOT / "configs" / "default.toml").read_text()) if modal.is_local() else {}

PROMPTS = {
    "fog": "a dashcam photo of the same road in thick fog, low visibility",
    "night": "a dashcam photo of the same road at night, street lights, dark sky",
    "rain": "a dashcam photo of the same road in heavy rain, wet road, grey sky",
    "snow": "a dashcam photo of the same road covered in snow, winter",
}
NEGATIVE = "cartoon, painting, text, watermark, distorted, extra cars, deformed"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.4.1", "diffusers==0.31.0", "transformers==4.46.3", "accelerate==1.1.1", "safetensors", "pillow",
                 "numpy<2", "huggingface_hub==0.26.2", "opencv-python-headless")
)
cache = modal.Volume.from_name("dgeo-hf-cache", create_if_missing=True)
app = modal.App("synthmix-generate", image=image)


@app.cls(gpu="A10G", volumes={"/root/.cache/huggingface": cache}, timeout=3600, scaledown_window=120,
         max_containers=10)
class Editor:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
        from transformers import pipeline

        self.depth = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0)
        cn_d = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
        cn_c = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=[cn_d, cn_c], torch_dtype=torch.float16,
            safety_checker=None).to("cuda")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        cache.commit()

    @modal.method()
    def edit(self, items: list[tuple[str, bytes, str, int]], out_w: int, out_h: int, g: dict) -> list[tuple[str, bytes]]:
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        res = []
        for name, jpg, edit, seed in items:
            im = Image.open(io.BytesIO(jpg)).convert("RGB")
            w, h = im.size
            W, H = g["gen_width"], int(round(g["gen_width"] * h / w / 8) * 8)
            im = im.resize((W, H), Image.LANCZOS)
            d = np.asarray(self.depth(im)["depth"].resize((W, H)), dtype=np.float32)
            d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
            depth_ctrl = Image.fromarray((np.stack([d] * 3, -1) * 255).astype("uint8"))
            gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
            canny_ctrl = Image.fromarray(np.stack([cv2.Canny(gray, 100, 200)] * 3, -1))
            gen = torch.Generator("cuda").manual_seed(seed)
            out = self.pipe(PROMPTS[edit], image=[depth_ctrl, canny_ctrl], negative_prompt=NEGATIVE,
                            num_inference_steps=g["steps"], guidance_scale=g["guidance"],
                            controlnet_conditioning_scale=[g["depth_scale"], g["canny_scale"]], generator=gen).images[0]
            out = out.resize((out_w, out_h), Image.LANCZOS)
            buf = io.BytesIO()
            out.save(buf, format="JPEG", quality=95)
            res.append((f"{name}__{edit}", buf.getvalue()))
        return res


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    import numpy as np
    import pandas as pd

    d, g = CFG["data"], CFG["generate"]
    splits = pd.read_csv(ROOT / d["out_dir"] / "splits.csv")
    train = splits[splits.split == "train"].sort_values("id")
    rng = np.random.default_rng(d["seed"] + 1)
    src = train.sample(g["n_source"], random_state=int(rng.integers(1 << 31))).sort_values("id").reset_index(drop=True)
    edits = np.array((g["edits"] * (len(src) // len(g["edits"]) + 1))[:len(src)])
    rng.shuffle(edits)
    src["edit"] = edits
    src["seed"] = rng.integers(0, 1 << 31, len(src))
    src[["id", "edit", "seed"]].to_csv(ROOT / d["out_dir"] / "synth_plan.csv", index=False)
    if limit:
        src = src.iloc[:limit]
    out_dir = ROOT / d["out_dir"] / "synth" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in src.itertuples() if not (out_dir / f"{r.id}__{r.edit}.jpg").exists()]
    print(f"{len(todo)} images to generate ({len(src) - len(todo)} already done)")
    bdd = Path(d["bdd_dir"]) if Path(d["bdd_dir"]).is_absolute() else ROOT / d["bdd_dir"]
    batches = [[(r.id, (bdd / "images" / f"{r.id}.jpg").read_bytes(), r.edit, int(r.seed)) for r in todo[i:i + g["batch"]]]
               for i in range(0, len(todo), g["batch"])]
    n = 0
    for res in Editor().edit.starmap([(b, d["width"], d["height"], g) for b in batches], order_outputs=False):
        for name, data in res:
            (out_dir / f"{name}.jpg").write_bytes(data)
            n += 1
        print(f"{n}/{len(todo)}", flush=True)
    print("done")
