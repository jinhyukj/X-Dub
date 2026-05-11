#!/usr/bin/env python3
"""Batch X-Dub inference: load the model ONCE, iterate over many videos.

Drop-in equivalent to running infer_lip_sync_pipeline.py once per video, but
the LipSyncPipeline is constructed exactly once and reused for every stem.
Saves ~30-60 sec of startup per video.

For each stem:
  - extract 16 kHz mono PCM audio from the source video (same as the bash driver)
  - preprocess_inputs → infer_one_sample (writes {stem}_output.mp4 + 2 compares
    into a per-stem tmp dir)
  - copy {stem}_output.mp4 to OUT_DIR/{stem}.mp4
  - cleanup tmp dir + audio file

Failures on one video are logged with [FAIL] and the loop continues.

Usage:
  python infer_lip_sync_batch.py \\
      --src_videos_dir /home/work/.local/HDTF/HDTF_original_long_vid/videos_cfr \\
      --output_dir /path/to/final/outputs \\
      --tmp_root /tmp \\
      --video_suffix _cfr25 \\
      --stems WDA_NancyPelosi0_000 WDA_TammyDuckworth_000 ... \\
      --ref_cfg_scale 2.5 --audio_cfg_scale 10.0 --num_inference_steps 30

Honors the X_DUB_DECODE_CHUNK env var (same as the single-video script).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback

import torch

# Reuse everything from the single-video script.
import infer_lip_sync_pipeline as base


def extract_audio(video_path, audio_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
         audio_path],
        check=True, capture_output=True,
    )


def build_sub_args(global_args, stem, src_video, audio_tmp, output_tmp):
    """Namespace passed into preprocess_inputs / infer_one_sample, mimicking
    what parse_args() would have produced for a single-video invocation."""
    return argparse.Namespace(
        video_path=src_video,
        audio_path=audio_tmp,
        sample_name=stem,
        ref_cfg_scale=global_args.ref_cfg_scale,
        audio_cfg_scale=global_args.audio_cfg_scale,
        audio_feat_window_size=global_args.audio_feat_window_size,
        num_inference_steps=global_args.num_inference_steps,
        seed=global_args.seed,
        ckpt_path=global_args.ckpt_path,
        output_dir=output_tmp,
        timing=global_args.timing,
        timing_csv=global_args.timing_csv,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", nargs="+", required=True,
                    help="Bare stems (no _cfr25 suffix). Output goes to <output_dir>/<stem>.mp4")
    ap.add_argument("--src_videos_dir", required=True,
                    help="Where to look for source mp4s. Filename = <stem><video_suffix>.mp4")
    ap.add_argument("--output_dir", required=True,
                    help="Where final per-stem mp4s land")
    ap.add_argument("--tmp_root", default="/tmp",
                    help="Where the per-stem tmp dirs (audio + intermediate mp4s) live")
    ap.add_argument("--video_suffix", default="_cfr25",
                    help="Suffix appended to stem to construct source filename")
    ap.add_argument("--ref_cfg_scale", type=float, default=2.5)
    ap.add_argument("--audio_cfg_scale", type=float, default=10.0)
    ap.add_argument("--audio_feat_window_size", type=int, default=0)
    ap.add_argument("--num_inference_steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_path", default=base.DEFAULT_DIT_PATH)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--timing_csv", default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_root, exist_ok=True)

    # Match the timing-globals dance in base.main() so infer_one_sample's
    # `if _TIMING_ENABLED:` branches see real values.
    base._TIMING_ENABLED = bool(args.timing)
    base._TIMING_CURRENT = {}
    base._TIMING_ROWS = []

    # Build pipe ONCE for the whole batch.
    pipe_args = argparse.Namespace(audio_feat_window_size=args.audio_feat_window_size)
    print(f"[batch] loading LipSyncPipeline once for {len(args.stems)} stems "
          f"(decode_chunk={os.environ.get('X_DUB_DECODE_CHUNK', '0')})", flush=True)
    t_load_start = time.perf_counter()
    pipe = base.LipSyncPipeline().from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            base.ModelConfig(path=args.ckpt_path, **base.vram_config),
            base.ModelConfig(path=base.DEFAULT_TEXT_ENCODER_PATH, **base.vram_config),
            base.ModelConfig(path=base.DEFAULT_VAE_PATH, **base.vram_config),
        ],
        tokenizer_config=base.ModelConfig(path=base.DEFAULT_TOKENIZER_PATH),
        args=pipe_args,
        whisper_ckpt_path=base.DEFAULT_WHISPER_PATH,
        wav2vec_ckpt_path=base.DEFAULT_WAV2VEC_PATH,
    )
    pipe.whisper_processor.to(dtype=torch.float32)
    pipe.wav2vec_processor.to(dtype=torch.float32)
    t_load = time.perf_counter() - t_load_start
    print(f"[batch] pipeline ready in {t_load:.1f}s", flush=True)

    n_done = n_skip = n_fail = 0
    for i, stem in enumerate(args.stems, 1):
        N = len(args.stems)
        src_video = os.path.join(args.src_videos_dir, f"{stem}{args.video_suffix}.mp4")
        audio_tmp = os.path.join(args.tmp_root, f"x_dub_{stem}.wav")
        output_tmp = os.path.join(args.tmp_root, f"x_dub_results_{stem}")
        final_path = os.path.join(args.output_dir, f"{stem}.mp4")

        if os.path.exists(final_path):
            print(f"[{i}/{N}] [SKIP] {stem} (output exists at {final_path})", flush=True)
            n_skip += 1
            continue
        if not os.path.exists(src_video):
            print(f"[{i}/{N}] [MISSING] {stem} ← {src_video}", flush=True)
            n_fail += 1
            continue

        print(f"[{i}/{N}] [START] {stem} at {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        t0 = time.perf_counter()
        try:
            shutil.rmtree(output_tmp, ignore_errors=True)
            os.makedirs(output_tmp, exist_ok=True)
            extract_audio(src_video, audio_tmp)

            sub_args = build_sub_args(args, stem, src_video, audio_tmp, output_tmp)
            base._TIMING_CURRENT = {"name": stem}
            sample = base.preprocess_inputs(src_video, audio_tmp, sub_args)
            base.infer_one_sample(pipe, sample, stem, sub_args)

            output_mp4 = os.path.join(output_tmp, f"{stem}_output.mp4")
            if not os.path.exists(output_mp4):
                cands = [f for f in os.listdir(output_tmp)
                         if f.endswith(".mp4") and not f.endswith("_compare.mp4")]
                if not cands:
                    cands = [f for f in os.listdir(output_tmp) if f.endswith(".mp4")]
                if not cands:
                    raise RuntimeError(f"no mp4 produced in {output_tmp}")
                output_mp4 = os.path.join(output_tmp, cands[0])
            shutil.copy(output_mp4, final_path)
            elapsed = time.perf_counter() - t0
            print(f"[{i}/{N}] [DONE] {stem} → {final_path} ({elapsed:.1f}s)", flush=True)
            n_done += 1

        except Exception as e:
            traceback.print_exc()
            elapsed = time.perf_counter() - t0
            print(f"[{i}/{N}] [FAIL] {stem}: {type(e).__name__}: {e} (after {elapsed:.1f}s)",
                  flush=True, file=sys.stderr)
            print(f"[{i}/{N}] [FAIL] {stem}: {type(e).__name__}: {e}", flush=True)
            n_fail += 1

        finally:
            shutil.rmtree(output_tmp, ignore_errors=True)
            try:
                os.remove(audio_tmp)
            except FileNotFoundError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n[batch] summary: done={n_done} skip={n_skip} fail={n_fail} "
          f"of {len(args.stems)}", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
