import argparse
import csv
import os
import time
from dataclasses import dataclass

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Timing instrumentation (opt-in via --timing).
# ---------------------------------------------------------------------------
_TIMING_ENABLED = False
_TIMING_CURRENT: dict = {}
_TIMING_ROWS: list = []

def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

from diffsynth.pipelines.lip_sync import LipSyncPipeline, ModelConfig
from lip_sync_preprocess import preprocess_video_with_dwpose
from utils import (
    blend_crop_video_with_ref,
    color_correction,
    concat_pil_videos_horizontally,
    get_total_length,
    make_pingpong_indices,
    paste_video_back,
    save_video_with_audio,
)

CLIP_NUM_FRAMES = 77
MOTION_NUM_FRAMES = 5

CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEFAULT_DIT_PATH = os.path.join(CHECKPOINTS_DIR, "X-Dub_model.safetensors")
DEFAULT_TEXT_ENCODER_PATH = os.path.join(CHECKPOINTS_DIR, "models_t5_umt5-xxl-enc-bf16.safetensors")
DEFAULT_VAE_PATH = os.path.join(CHECKPOINTS_DIR, "Wan2.2_VAE.safetensors")
DEFAULT_TOKENIZER_PATH = os.path.join(CHECKPOINTS_DIR, "umt5-xxl")
DEFAULT_WHISPER_PATH = os.path.join(CHECKPOINTS_DIR, "whisper", "large-v2.pt")
DEFAULT_WAV2VEC_PATH = os.path.join(CHECKPOINTS_DIR, "wav2vec2-base-960h")

vram_config = {
    # "offload_dtype": "disk",
    # "offload_device": "disk",
    "offload_dtype": torch.bfloat16,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16,
    "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

@dataclass
class PreprocessedLipSyncSample:
    raw_video: list[Image.Image]
    ref_video: list[Image.Image]
    bboxes: list[list[int]]
    audio_path: str


def smooth_transition_latent(previous_segment, current_segment):
    boundary_latent = 0.5 * (previous_segment[:, :, -1:, :, :] + current_segment[:, :, 1:2, :, :])
    previous_segment = previous_segment.clone()
    previous_segment[:, :, -1:, :, :] = boundary_latent
    return previous_segment, current_segment[:, :, 2:, :, :]


def preprocess_inputs(video_path: str, audio_path: str, args) -> PreprocessedLipSyncSample:
    sample_name = args.sample_name or os.path.splitext(os.path.basename(video_path))[0]
    # ============================================================
    # STAGE 1: Video Ingestion + Face Detection + Crop (CPU/GPU)
    # ============================================================
    # Reads all video frames, runs DWPose (YOLOX + RTMPose) for facial landmarks,
    # smooths landmark trajectories (Savitzky-Golay), builds per-frame face bboxes,
    # then crops and resizes each frame to 512x512 reference faces.
    raw_video, ref_video, bboxes, case_flag = preprocess_video_with_dwpose(
        video_path,
        output_dir=args.output_dir,
        sample_name=sample_name,
    )
    print(
        f"[Preprocess {sample_name}] "
        f"num_raw_frames={len(raw_video)}, num_ref_frames={len(ref_video)}, case_flag={case_flag}"
    )
    print(f"[Preprocess {sample_name}] first_bbox={bboxes[0]}")
    return PreprocessedLipSyncSample(
        raw_video=raw_video,
        ref_video=ref_video,
        bboxes=bboxes,
        audio_path=audio_path,
    )


def validate_preprocessed_sample(sample: PreprocessedLipSyncSample):
    assert os.path.exists(sample.audio_path), f"audio path not found: {sample.audio_path}"
    assert len(sample.ref_video) > 0, "ref_video cannot be empty."
    assert len(sample.raw_video) == len(sample.ref_video), "raw_video and ref_video must have the same length."
    assert len(sample.bboxes) == len(sample.ref_video), "bboxes and ref_video must have the same length."
    for bbox in sample.bboxes:
        assert len(bbox) == 4, f"bbox must have 4 ints, got: {bbox}"
    return sample


def infer_one_sample(pipe, sample: PreprocessedLipSyncSample, sample_name: str, args):
    sample = validate_preprocessed_sample(sample)
    # Reset per-sample timing state so it doesn't carry over from a previous sample
    if hasattr(pipe, "_e2d_t0"):
        del pipe._e2d_t0
    raw_video = sample.raw_video
    ref_video = sample.ref_video
    bboxes = sample.bboxes
    audio_path = sample.audio_path

    print(f"[Sample {sample_name}] video_path={args.video_path}")
    print(f"[Sample {sample_name}] audio_path={audio_path}")
    print(f"[Sample {sample_name}] num_ref_frames={len(ref_video)}")

    # ============================================================
    # STAGE 2: Audio Length + Pingpong Indexing (CPU)
    # ============================================================
    # Reads audio duration, computes total output length and # of AR clips (77 frames
    # each with 5-frame motion overlap). Pingpong-folds the video if audio is longer
    # than the input video so ref/raw/bbox arrays match total_length.
    total_length = get_total_length(audio_path, clip_num_frames=CLIP_NUM_FRAMES, motion_num_frames=MOTION_NUM_FRAMES)
    num_clips = 1 + max(0, total_length - CLIP_NUM_FRAMES) // (CLIP_NUM_FRAMES - MOTION_NUM_FRAMES)
    pingpong_indices = make_pingpong_indices(len(ref_video), total_length)
    ref_video = [ref_video[index] for index in pingpong_indices]
    raw_video = [raw_video[index] for index in pingpong_indices]
    bboxes = [bboxes[index] for index in pingpong_indices]

    print(f"[Sample {sample_name}] total_length={total_length}, num_clips={num_clips}")

    motion_video = None
    latents_segments = []
    start_idx = 0

    # ============================================================
    # STAGE 3a (hoisted): Whisper + Wav2Vec audio encode (GPU) — once for whole audio
    # ============================================================
    # Pre-load BOTH audio_encoder and vae before any timing starts, so that VRAM-swap
    # overhead doesn't inflate audio_to_decode or encode_to_decode measurements
    # (other models load all weights upfront, so this matches their behaviour).
    pipe.load_models_to_device(["audio_encoder", "vae"])

    # _a2d_t0: start of audio_to_decode span (Def 1)
    if _TIMING_ENABLED:
        _gpu_sync()
    _a2d_t0 = time.perf_counter()

    print(f"[Sample {sample_name}] Encoding whisper features for full audio ...")
    whisper_feat = pipe.whisper_processor.audio2feat(audio_path, use_cache=True)
    print(f"[Sample {sample_name}] Encoding wav2vec features for full audio ...")
    wav2vec_feat = pipe.wav2vec_processor.audio2feat(audio_path, use_cache=True)

    # ============================================================
    # STAGE 3: AR Clip Loop (GPU) — repeats num_clips times
    # ============================================================
    # Each iteration processes CLIP_NUM_FRAMES (77) frames, using MOTION_NUM_FRAMES (5)
    # motion frames from the previous clip for temporal consistency. The pipe() call
    # internally runs stages 3a-3d below.
    for clip_idx in range(num_clips):
        print(f"[Sample {sample_name}] [{clip_idx + 1}/{num_clips}] start_idx={start_idx}")
        ref_video_clip = ref_video[start_idx: start_idx + CLIP_NUM_FRAMES]
        # _enc_to_dec_t0: start of encode_to_decode span (Def 2) — set just before
        # clip 0's pipe() so we capture the first VAE encode start.
        if clip_idx == 0:
            if _TIMING_ENABLED:
                _gpu_sync()
            _enc_to_dec_t0 = time.perf_counter()
        # ---------------------------------------------------------
        #   Inside pipe.__call__:
        #   (audio features already encoded above; here we just slice per-clip windows)
        #   STAGE 3b: T5 umt5-xxl text encode (GPU, empty prompt)
        #   STAGE 3c: VAE encode ref_video_clip + motion frames (GPU)
        #   STAGE 3d: DiT diffusion denoise, 30 steps with dynamic CFG (GPU)
        # ---------------------------------------------------------
        output_video, outputs = pipe(
            ref_video=ref_video_clip,
            start_idx=start_idx,
            audio_npy_path=None,
            audio_wav_path=audio_path,
            whisper_feat=whisper_feat,
            wav2vec_feat=wav2vec_feat,
            prompt="",
            motion_video=motion_video,
            height=512,
            width=512,
            num_frames=CLIP_NUM_FRAMES,
            motion_latents_num_frames=2,
            ref_cfg_scale=args.ref_cfg_scale,
            audio_cfg_scale=args.audio_cfg_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            use_dynamic_cfg=True,
            replace_border_latents=True,
            replace_border_latents_width=1,
        )

        if whisper_feat is None:
            whisper_feat = outputs.get("whisper_feat", None)
        if wav2vec_feat is None:
            wav2vec_feat = outputs.get("wav2vec_feat", None)

        # ---------------------------------------------------------
        # STAGE 3e: Color correction + motion frame extraction (CPU/GPU)
        # ---------------------------------------------------------
        output_video = color_correction(output_video, ref_video_clip)
        motion_video = output_video[-MOTION_NUM_FRAMES:]
        output_latents = outputs["latents"]

        # ---------------------------------------------------------
        # STAGE 3f: Inter-clip latent smoothing (GPU)
        # ---------------------------------------------------------
        if clip_idx == 0:
            latents_segments.append(output_latents)
        else:
            latents_segments[-1], output_latents_to_append = smooth_transition_latent(latents_segments[-1], output_latents)
            latents_segments.append(output_latents_to_append)

        start_idx += CLIP_NUM_FRAMES - MOTION_NUM_FRAMES


    # ============================================================
    # STAGE 4: Final VAE Decode (GPU) — decode all accumulated latents
    # ============================================================
    # Single decode call across the full latent sequence. Peak GPU memory is
    # bounded inside VAE.decode() by per-frame CPU accumulation of the output
    # tensor (see VideoVAE38_.decode in diffsynth/models/wan_video_vae.py),
    # so this no longer OOMs on long videos. Returned tensor is CPU-resident.
    output_latents = torch.cat(latents_segments, dim=2)
    pipe.load_models_to_device(["vae"])
    final_video = pipe.vae.decode(output_latents, device=pipe.device, tiled=False, tile_size=(32, 32), tile_stride=(16, 16))
    final_output_video = pipe.vae_output_to_video(final_video)[:total_length]
    final_output_video = color_correction(final_output_video, ref_video[:total_length])

    # ============================================================
    # End of audio_to_decode (Def 1), encode_to_decode (Def 2 broad),
    # and pure_encode_to_decode (Def 2 narrow) spans
    # ============================================================
    if _TIMING_ENABLED:
        _gpu_sync()
        _now = time.perf_counter()
        _TIMING_CURRENT["audio_to_decode"] = _now - _a2d_t0
        _TIMING_CURRENT["encode_to_decode"] = _now - _enc_to_dec_t0
        # pipe._e2d_t0 was set inside WanVideoUnit_ReferenceVideoEmbedder
        # right before the FIRST pipe.vae.encode() call on clip 0.
        if hasattr(pipe, "_e2d_t0"):
            _TIMING_CURRENT["pure_encode_to_decode"] = _now - pipe._e2d_t0

    # ============================================================
    # STAGE 5: Border Blending (CPU) — blend generated crop with reference borders
    # ============================================================
    # Replaces a narrow border region of the generated crop with the original reference
    # crop pixels to reduce visible seams at the crop boundary.
    blended_crop_video = blend_crop_video_with_ref(
        final_output_video,
        ref_video[:total_length],
        replace_border_latents_width=1,
    )
    # ============================================================
    # STAGE 6: Paste-back (CPU) — warp blended crops onto full-resolution raw frames
    # ============================================================
    # Uses the per-frame bboxes from Stage 1 to paste the 512x512 generated face region
    # back into the original full-resolution frames.
    final_pasted_video = paste_video_back(blended_crop_video, raw_video, bboxes)

    # ============================================================
    # STAGE 7: Video Write + Audio Mux (CPU)
    # ============================================================
    # Save standalone pasted video (just the output, no side-by-side comparison)
    save_video_with_audio(final_pasted_video, audio_path, f"{sample_name}_output", args.output_dir)

    crop_compare_video = concat_pil_videos_horizontally(ref_video[:total_length], final_output_video)
    save_video_with_audio(crop_compare_video, audio_path, f"{sample_name}_crop_compare", args.output_dir)

    paste_compare_video = concat_pil_videos_horizontally(raw_video[:total_length], final_pasted_video)
    save_video_with_audio(paste_compare_video, audio_path, f"{sample_name}_paste_compare", args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--sample_name", type=str, default=None)
    parser.add_argument("--ref_cfg_scale", type=float, default=2.5)
    parser.add_argument("--audio_cfg_scale", type=float, default=10.0)
    parser.add_argument("--audio_feat_window_size", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_DIT_PATH)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--timing", action="store_true",
                        help="Enable timing measurements (audio_to_decode, encode_to_decode).")
    parser.add_argument("--timing_csv", type=str, default=None,
                        help="Path to write timing CSV. Default: <output_dir>/timing.csv")
    return parser.parse_args()


def main():
    global _TIMING_ENABLED, _TIMING_CURRENT, _TIMING_ROWS
    args = parse_args()
    _TIMING_ENABLED = bool(args.timing)
    _TIMING_CURRENT = {"name": args.sample_name or os.path.splitext(os.path.basename(args.video_path))[0]}
    sample_name = args.sample_name or os.path.splitext(os.path.basename(args.video_path))[0]
    sample = preprocess_inputs(args.video_path, args.audio_path, args)

    pipe = LipSyncPipeline().from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=args.ckpt_path, **vram_config),
            ModelConfig(path=DEFAULT_TEXT_ENCODER_PATH, **vram_config),
            ModelConfig(path=DEFAULT_VAE_PATH, **vram_config),
        ],
        tokenizer_config=ModelConfig(path=DEFAULT_TOKENIZER_PATH),
        args=args,
        whisper_ckpt_path=DEFAULT_WHISPER_PATH,
        wav2vec_ckpt_path=DEFAULT_WAV2VEC_PATH,
    )
    pipe.whisper_processor.to(dtype=torch.float32)
    pipe.wav2vec_processor.to(dtype=torch.float32)
    infer_one_sample(pipe, sample, sample_name, args)

    # Write timing CSV
    if _TIMING_ENABLED and _TIMING_CURRENT:
        peak_alloc = peak_reserved = 0.0
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9
            _TIMING_CURRENT["peak_alloc_gb"] = peak_alloc
            _TIMING_CURRENT["peak_reserved_gb"] = peak_reserved
        csv_path = args.timing_csv or os.path.join(args.output_dir, "timing.csv")
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
        fieldnames = ["name", "audio_to_decode", "encode_to_decode",
                      "pure_encode_to_decode", "peak_alloc_gb", "peak_reserved_gb"]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            row = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in _TIMING_CURRENT.items()}
            writer.writerow(row)
        print(f"\n[Timing] {_TIMING_CURRENT}")
        print(f"[Timing] appended row to {csv_path}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        print(
            f"[VRAM] peak_allocated={peak_gb:.2f} GB peak_reserved={reserved_gb:.2f} GB",
            flush=True,
        )


if __name__ == "__main__":
    main()
