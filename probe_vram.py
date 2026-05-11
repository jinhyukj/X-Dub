"""Single-clip VRAM probe for X-Dub on H100."""
import os
import pathlib
import re
import subprocess

import modal

_LOCAL_XDUB_ROOT = pathlib.Path("/home/work/.local/X-Dub")

XDUB_VOL_NAME = "xdub-input"
FG_VOL_NAME = "fastgen-assets"
xdub_volume = modal.Volume.from_name(XDUB_VOL_NAME, create_if_missing=False)
fg_volume = modal.Volume.from_name(FG_VOL_NAME, create_if_missing=False)

ASSETS = "/assets"
FG_MOUNT = "/assets-fast"
CKPTS = f"{ASSETS}/xdub_ckpts"
TEST_CLIPS_DIR = f"{FG_MOUNT}/hdtf/videos_batch"

CLIP = "WDA_BarackObama_000_cfr25.mp4"

_ignore = [
    ".git/**", "__pycache__/**", "*.pyc",
    "checkpoints/**", "results/**", "*.mp4", "*.wav",
    "assets/**",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10",
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0",
                 "build-essential", "g++", "clang", "ninja-build")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "diffusers==0.37.1", "transformers==4.50.0", "accelerate",
        "safetensors==0.7.0", "sentencepiece==0.2.1", "protobuf==6.33.5",
        "modelscope==1.34.0", "huggingface_hub==0.34.0", "ftfy==6.3.1",
        "pandas==2.3.3", "imageio==2.37.2", "imageio-ffmpeg==0.6.0",
        "ffmpeg-python==0.2.0", "einops==0.8.2", "numpy==1.26.4",
        "scipy==1.15.3", "pillow==10.4.0", "tqdm==4.67.3",
        "regex", "opencv-python==4.10.0.84", "typing_extensions==4.15.0",
        "peft", "datasets",
    )
    .pip_install("setuptools", "wheel")
    .pip_install("mmengine==0.10.7")
    .run_commands(
        "MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST='9.0' "
        "pip install mmcv==2.1.0 --no-build-isolation"
    )
    .pip_install("mmdet==3.2.0", "mmpose==1.3.2")
    .add_local_dir(str(_LOCAL_XDUB_ROOT), remote_path="/workspace/X-Dub",
                   ignore=_ignore, copy=True)
)

app = modal.App("x-dub-vram-probe", image=image)


@app.function(gpu="H100", volumes={ASSETS: xdub_volume, FG_MOUNT: fg_volume},
              timeout=60 * 30)
def probe_one() -> str:
    import shutil

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = "/workspace/X-Dub:" + env.get("PYTHONPATH", "")

    work_dir = "/workspace/X-Dub"
    ckpt_link = f"{work_dir}/checkpoints"
    if not os.path.exists(ckpt_link):
        os.symlink(CKPTS, ckpt_link)

    dwpose_dst = f"{work_dir}/dwpose_tools/models"
    dwpose_src = f"{CKPTS}/dwpose_tools/models"
    if os.path.exists(dwpose_src):
        os.makedirs(dwpose_dst, exist_ok=True)
        for f in os.listdir(dwpose_src):
            dst = f"{dwpose_dst}/{f}"
            if not os.path.exists(dst):
                shutil.copy2(f"{dwpose_src}/{f}", dst)

    stem = CLIP.replace("_cfr25.mp4", "")
    out_dir = f"/tmp/x_dub_out/{stem}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/{stem}.timing.csv"

    cmd = [
        "python", "infer_lip_sync_pipeline.py",
        "--video_path", f"{TEST_CLIPS_DIR}/{CLIP}",
        "--audio_path", f"{TEST_CLIPS_DIR}/{CLIP}",
        "--ckpt_path", f"{CKPTS}/X-Dub_model.safetensors",
        "--ref_cfg_scale", "2.5", "--audio_cfg_scale", "10.0",
        "--num_inference_steps", "30",
        "--output_dir", out_dir,
        "--timing", "--timing_csv", csv_path,
    ]
    result = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True)
    out = result.stdout + "\n" + result.stderr
    # Print full stdout so we can see all events (probe lines + ffmpeg + errors).
    print("=== STDOUT ===")
    print(result.stdout)
    print("=== STDERR (tail) ===")
    print(result.stderr[-2000:])
    m = re.search(r"\[VRAM-PROBE\]\s*peak_allocated=(.*)", out)
    return m.group(0) if m else "no VRAM-PROBE line found"


@app.local_entrypoint()
def probe():
    print(f"[X-Dub] probing VRAM on H100 with clip {CLIP} ...")
    line = probe_one.remote()
    print(f"\nX-Dub: {line}")
