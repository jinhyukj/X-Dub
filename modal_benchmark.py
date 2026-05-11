"""Modal benchmark: run X-Dub on 10 HDTF test clips.

Usage:
    modal run modal_benchmark.py::upload_checkpoints  # one-time, uploads ckpts to volume
    modal run modal_benchmark.py::sweep               # run all 10 clips
"""
import os
import pathlib
import subprocess
import modal

_LOCAL_XDUB_ROOT = pathlib.Path("/home/work/.local/X-Dub")

VOL = modal.Volume.from_name("v2v-benchmark", create_if_missing=True)
ASSETS = "/assets"
CKPTS = f"{ASSETS}/xdub_ckpts"

WARMUP_CLIP = "WDA_AndyLevin_000_cfr25.mp4"
CLIPS = [
    "RD_Radio18_000_cfr25.mp4", "WDA_BarackObama_000_cfr25.mp4",
    "WDA_DonnaShalala1_000_cfr25.mp4", "WDA_JerryNadler_000_cfr25.mp4",
    "WDA_KatherineClark_000_cfr25.mp4", "WDA_NancyPelosi0_000_cfr25.mp4",
    "WDA_TedLieu_000_cfr25.mp4", "WRA_AdamKinzinger1_000_cfr25.mp4",
    "WRA_EricCantor_000_cfr25.mp4", "WRA_MittRomney_000_cfr25.mp4",
]

_ignore = [
    ".git/**", "__pycache__/**", "*.pyc",
    "checkpoints/**", "results/**", "*.mp4", "*.wav",
    "assets/**",
]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", add_python="3.10",
    )
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "build-essential", "g++", "clang", "ninja-build")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cu124",
    )
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
    .add_local_dir(
        str(_LOCAL_XDUB_ROOT), remote_path="/workspace/X-Dub",
        ignore=_ignore, copy=True,
    )
)

app = modal.App("x-dub-benchmark", image=image)


@app.function(volumes={ASSETS: VOL}, timeout=60 * 60)
def upload_checkpoints():
    """Placeholder — upload checkpoints via `modal volume put` instead.

    modal volume put v2v-benchmark /home/work/.local/X-Dub/checkpoints/ xdub_ckpts/
    """
    print("Use: modal volume put v2v-benchmark /home/work/.local/X-Dub/checkpoints/ xdub_ckpts/")
    VOL.commit()


@app.function(gpu="H200", volumes={ASSETS: VOL}, timeout=60 * 30, max_containers=1)
def run_benchmark(video_name: str) -> str:
    """Run X-Dub on one clip."""
    import shutil

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = "/workspace/X-Dub:" + env.get("PYTHONPATH", "")

    work_dir = "/workspace/X-Dub"
    # Symlink checkpoints from volume to the expected path
    ckpt_link = f"{work_dir}/checkpoints"
    if not os.path.exists(ckpt_link):
        os.symlink(CKPTS, ckpt_link)

    # Copy DWPose models to expected location
    dwpose_dst = f"{work_dir}/dwpose_tools/models"
    dwpose_src = f"{CKPTS}/dwpose_tools/models"
    if os.path.exists(dwpose_src) and not os.listdir(dwpose_dst) if os.path.exists(dwpose_dst) else True:
        os.makedirs(dwpose_dst, exist_ok=True)
        for f in os.listdir(dwpose_src):
            dst = f"{dwpose_dst}/{f}"
            if not os.path.exists(dst):
                shutil.copy2(f"{dwpose_src}/{f}", dst)

    stem = video_name.replace("_cfr25.mp4", "")
    out_dir = f"/tmp/x_dub_out/{stem}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", "infer_lip_sync_pipeline.py",
        "--video_path", f"{ASSETS}/test_clips/{video_name}",
        "--audio_path", f"{ASSETS}/test_clips/{video_name}",
        "--ckpt_path", f"{CKPTS}/X-Dub_model.safetensors",
        "--ref_cfg_scale", "2.5",
        "--audio_cfg_scale", "10.0",
        "--num_inference_steps", "30",
        "--output_dir", out_dir,
    ]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    # Find the standalone output mp4 (not the _compare versions)
    result_mp4 = None
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith("_output.mp4"):
                result_mp4 = os.path.join(root, f)
                break
        if result_mp4:
            break

    if result_mp4:
        vol_dir = f"{ASSETS}/output_videos/x_dub"
        os.makedirs(vol_dir, exist_ok=True)
        shutil.copy2(result_mp4, f"{vol_dir}/{stem}.mp4")
        VOL.commit()
        return "OK"
    return "FAIL: no output mp4"


@app.local_entrypoint()
def sweep():
    print(f"Running {len(CLIPS)} clips on X-Dub (with 1 warmup)")
    all_clips = [WARMUP_CLIP] + CLIPS
    for i, vn in enumerate(all_clips):
        stem = vn.replace("_cfr25.mp4", "")
        try:
            result = run_benchmark.remote(vn)
            tag = "WARMUP" if i == 0 else "OK"
            print(f"  [{tag}] {stem}: {result}")
        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
    print("Done! Videos at output_videos/x_dub/ on the volume.")
