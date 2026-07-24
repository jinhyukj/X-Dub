#!/usr/bin/env bash
# X-Dub long-video inference, 4-GPU parallel.
# Generates only stems with NO existing output (symlinks count as present).
set -u
cd /home/work/.local/X-Dub
source ~/.local/miniconda3/etc/profile.d/conda.sh
conda activate x-dub

SRC=/home/work/.local/HDTF/HDTF_original_long_vid/videos_cfr
OUT=/home/work/.local/stitch_sources/hdtf_long/xdub
SUFFIX=_cfr25
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=/home/work/.local/X-Dub/long_4gpu_logs_$TS
mkdir -p "$LOGDIR" "$OUT"

# 1) collect missing stems (-e resolves symlink, so existing old outputs are skipped)
missing=()
for f in "$SRC"/*"${SUFFIX}".mp4; do
  base=$(basename "$f"); stem=${base%${SUFFIX}.mp4}
  [ -e "$OUT/$stem.mp4" ] || missing+=("$stem")
done
echo "[launcher] missing ${#missing[@]} stems -> $LOGDIR" | tee "$LOGDIR/launcher.log"
printf "%s\n" "${missing[@]}" >> "$LOGDIR/launcher.log"

# 2) round-robin split into 4 groups, one nohup process per GPU
for g in 0 1 2 3; do
  group=()
  for ((i=g; i<${#missing[@]}; i+=4)); do group+=("${missing[$i]}"); done
  [ ${#group[@]} -eq 0 ] && continue
  echo "[launcher] GPU $g (${#group[@]}): ${group[*]}" | tee -a "$LOGDIR/launcher.log"
  CUDA_VISIBLE_DEVICES=$g X_DUB_DECODE_CHUNK=0 \
    nohup python infer_lip_sync_batch.py \
      --src_videos_dir "$SRC" --output_dir "$OUT" \
      --tmp_root /tmp/xdub_gpu$g --video_suffix "$SUFFIX" \
      --ref_cfg_scale 2.5 --audio_cfg_scale 10.0 --num_inference_steps 30 \
      --stems "${group[@]}" \
      > "$LOGDIR/gpu${g}.log" 2>&1 &
  echo $! > "$LOGDIR/gpu${g}.pid"
  echo "[launcher] GPU $g pid $(cat $LOGDIR/gpu${g}.pid)" | tee -a "$LOGDIR/launcher.log"
done
echo "$LOGDIR" > /home/work/.local/X-Dub/.long_4gpu_latest
echo "[launcher] done. tail -f $LOGDIR/gpu0.log"
