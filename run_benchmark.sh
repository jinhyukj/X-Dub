#!/bin/bash
# Run X-Dub on 10 HDTF test clips.
# Usage: CUDA_VISIBLE_DEVICES=0 bash run_benchmark.sh
set -euo pipefail

PYTHON="/home/work/.local/miniconda3/envs/x-dub/bin/python"
SCRIPT="/home/work/.local/X-Dub/infer_lip_sync_pipeline.py"
TEST_CLIPS="/home/work/.local/HDTF/HDTF_testset_10clips"
OUT_DIR="/home/work/.local/hyunbin/FastGen-redmd/modal_out/output_videos/x_dub"

mkdir -p "$OUT_DIR"

cd /home/work/.local/X-Dub

for clip in "$TEST_CLIPS"/*_cfr25.mp4; do
    stem=$(basename "$clip" _cfr25.mp4)
    out_path="${OUT_DIR}/${stem}.mp4"

    if [ -f "$out_path" ]; then
        echo "[SKIP] $stem (already exists)"
        continue
    fi

    echo "========================================"
    echo "Processing: $stem"
    echo "========================================"

    # Extract audio from clip
    audio_tmp="/tmp/x_dub_${stem}.wav"
    ffmpeg -y -i "$clip" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$audio_tmp" 2>/dev/null

    $PYTHON "$SCRIPT" \
        --video_path "$clip" \
        --audio_path "$audio_tmp" \
        --ref_cfg_scale 2.5 \
        --audio_cfg_scale 10.0 \
        --num_inference_steps 30 \
        --output_dir "/tmp/x_dub_results_${stem}"

    # Find the output video (X-Dub saves to output_dir with auto-naming)
    result=$(find "/tmp/x_dub_results_${stem}" -name "*.mp4" | head -1)
    if [ -n "$result" ]; then
        cp "$result" "$out_path"
        echo "[OK] $stem -> $out_path"
    else
        echo "[FAIL] $stem - no output found"
    fi

    rm -f "$audio_tmp"
    rm -rf "/tmp/x_dub_results_${stem}"
done

echo "Done! Results in $OUT_DIR"
