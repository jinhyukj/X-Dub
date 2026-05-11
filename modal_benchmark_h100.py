"""X-Dub benchmark on H100: 5 HDTF clips, 3 timing definitions.

Volumes:
    fastgen-assets - shared HDTF clips (read-only) at /assets-fast/hdtf/videos_batch/
    xdub-input     - X-Dub checkpoints + dwpose models at /assets/xdub_ckpts/

Usage:
    modal run modal_benchmark_h100.py::upload_checkpoints   # one-time
    modal run modal_benchmark_h100.py::sweep
"""
import os
import pathlib
import subprocess
import modal

_LOCAL_XDUB_ROOT = pathlib.Path("/home/work/.local/X-Dub")

XDUB_VOL_NAME = "xdub-input"
FG_VOL_NAME = "fastgen-assets"
xdub_volume = modal.Volume.from_name(XDUB_VOL_NAME, create_if_missing=True)
fg_volume = modal.Volume.from_name(FG_VOL_NAME, create_if_missing=False)

ASSETS = "/assets"
FG_MOUNT = "/assets-fast"
CKPTS = f"{ASSETS}/xdub_ckpts"
TEST_CLIPS_DIR = f"{FG_MOUNT}/hdtf/videos_batch"

CLIPS = [
    "WDA_BarackObama_000_cfr25.mp4",
    "WDA_DonnaShalala1_000_cfr25.mp4",
    "WDA_NancyPelosi0_000_cfr25.mp4",
    "WRA_AdamKinzinger1_000_cfr25.mp4",
    "WRA_MittRomney_000_cfr25.mp4",
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
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0",
                 "build-essential", "g++", "clang", "ninja-build")
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

app = modal.App("x-dub-benchmark-h100", image=image)


@app.function(
    gpu="H100",
    volumes={ASSETS: xdub_volume, FG_MOUNT: fg_volume},
    timeout=60 * 30,
    max_containers=1,
)
def run_benchmark(video_name: str) -> str:
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

    stem = video_name.replace("_cfr25.mp4", "")
    out_dir = f"/tmp/x_dub_out/{stem}"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/{stem}.timing.csv"

    cmd = [
        "python", "infer_lip_sync_pipeline.py",
        "--video_path", f"{TEST_CLIPS_DIR}/{video_name}",
        "--audio_path", f"{TEST_CLIPS_DIR}/{video_name}",
        "--ckpt_path", f"{CKPTS}/X-Dub_model.safetensors",
        "--ref_cfg_scale", "2.5",
        "--audio_cfg_scale", "10.0",
        "--num_inference_steps", "30",
        "--output_dir", out_dir,
        "--timing",
        "--timing_csv", csv_path,
    ]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir, env=env, check=True)

    # Persist output video
    result_mp4 = None
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith("_output.mp4"):
                result_mp4 = os.path.join(root, f)
                break
        if result_mp4:
            break
    if result_mp4:
        vol_dir = f"{ASSETS}/bench_h100_5clips/output_videos/x_dub"
        os.makedirs(vol_dir, exist_ok=True)
        shutil.copy2(result_mp4, f"{vol_dir}/{stem}.mp4")
        xdub_volume.commit()

    if os.path.exists(csv_path):
        return pathlib.Path(csv_path).read_text()
    return ""


@app.local_entrypoint()
def sweep():
    print(f"[x-dub] Running {len(CLIPS)} clips on H100 (max_containers=1).")

    out_dir = _LOCAL_XDUB_ROOT / "modal_out_h100_5clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for vn in CLIPS:
        stem = vn.replace("_cfr25.mp4", "")
        try:
            csv_text = run_benchmark.remote(vn)
        except Exception as e:
            print(f"  [FAIL] {stem}: {e}")
            continue
        if csv_text:
            (out_dir / f"{stem}.timing.csv").write_text(csv_text)
            lines = csv_text.strip().splitlines()
            if len(lines) >= 2:
                rows.append(lines)
            print(f"  [OK]   {stem}")
        else:
            print(f"  [FAIL] {stem}: empty CSV")

    if rows:
        agg_path = out_dir / "aggregate.csv"
        with open(agg_path, "w") as f:
            f.write(rows[0][0] + "\n")
            for r in rows:
                for line in r[1:]:
                    if line and not line.startswith("name") and "AVERAGE" not in line:
                        f.write(line + "\n")
        print(f"\nAggregate: {agg_path}")
    print("Done: x-dub")


@app.function(
    image=modal.Image.debian_slim(),
    volumes={ASSETS: xdub_volume},
    timeout=3600,
)
def _upload_one(file_name: str, data: bytes, dest_path: str):
    p = pathlib.Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    print(f"  -> {dest_path} ({len(data) / 1e6:.1f} MB)")
    xdub_volume.commit()


@app.local_entrypoint()
def upload_checkpoints():
    """Upload X-Dub checkpoints + dwpose tools to /assets/xdub_ckpts/."""
    todo = [
        (pathlib.Path("/home/work/.local/X-Dub/checkpoints"), "xdub_ckpts"),
        (pathlib.Path("/home/work/.local/X-Dub/dwpose_tools/models"),
         "xdub_ckpts/dwpose_tools/models"),
    ]
    for local_root, remote_subdir in todo:
        if not local_root.exists():
            print(f"Local missing: {local_root}")
            continue
        files = sorted(f for f in local_root.rglob("*") if f.is_file())
        total_gb = sum(f.stat().st_size for f in files) / 1e9
        print(f"\n[{local_root.name}] {len(files)} files, {total_gb:.2f} GB")
        for local_file in files:
            rel = local_file.relative_to(local_root)
            dest_path = f"{ASSETS}/{remote_subdir}/{rel}"
            size_mb = local_file.stat().st_size / 1e6
            print(f"Uploading {rel} ({size_mb:.1f} MB)")
            _upload_one.remote(local_file.name, local_file.read_bytes(), dest_path)
    print("\nAll uploads complete.")
