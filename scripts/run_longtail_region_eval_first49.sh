#!/usr/bin/env bash
set -euo pipefail

ROOT="/ssd2/wenyan/CarTwin/longtail/PVG"
SOURCE_PATH="$ROOT/data_longtail/0c1afba4-e796-43c6-ba36-6a225b6f2968"
MODEL_PATH="$ROOT/eval_output/longtail_v2_first49/0c1afba4-e796-43c6-ba36-6a225b6f2968"
OUTPUT_ROOT="$ROOT/eval_output/longtail_region_eval_first49"
SCENE_NAME="0c1afba4-e796-43c6-ba36-6a225b6f2968"
GPU_ID="${1:-6}"

cd "$ROOT"

mkdir -p "$OUTPUT_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_longtail_regions.py \
  --config configs/waymo_reconstruction.yaml \
  source_path="$SOURCE_PATH" \
  model_path="$MODEL_PATH" \
  start_frame=0 \
  end_frame=48 \
  enable_long_tail_branch=true \
  cam_num=3

LATEST_ITER=$(find "$MODEL_PATH" -maxdepth 1 -type f -name 'chkpnt*.pth' | sed 's/.*chkpnt\([0-9]\+\)\.pth/\1/' | sort -n | tail -n 1)
if [[ -z "$LATEST_ITER" ]]; then
  echo "No checkpoint found in $MODEL_PATH" >&2
  exit 1
fi

DEST_DIR="$OUTPUT_ROOT/${SCENE_NAME}_iter${LATEST_ITER}"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

cp "$MODEL_PATH/eval/region_metrics_${LATEST_ITER}.json" "$DEST_DIR/"
cp "$MODEL_PATH/eval/region_metrics_${LATEST_ITER}.md" "$DEST_DIR/"
cp -r "$MODEL_PATH/eval/test_${LATEST_ITER}_region_metrics" "$DEST_DIR/"
cp -r "$MODEL_PATH/eval/train_${LATEST_ITER}_region_metrics" "$DEST_DIR/"

cat <<MSG
Long-tail region evaluation finished.
Saved results to:
  $DEST_DIR
MSG
