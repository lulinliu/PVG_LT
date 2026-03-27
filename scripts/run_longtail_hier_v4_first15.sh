#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="/ssd2/wenyan/code/longtail/PVG_LT"
DATA_ROOT="/ssd2/wenyan/code/longtail/longtail_data"
SCENE_ID="0c1afba4-e796-43c6-ba36-6a225b6f2968"
DATA_PATH="$DATA_ROOT/data_longtail/$SCENE_ID"
GPU_ID="${1:-0}"
TASK="${2:-recon_full}"
RUN_TAG="${3:-}"
START_FRAME=0
END_FRAME=14

RECON_CONFIG="configs/waymo_reconstruction_longtail_hier_v4_first15.yaml"
RECON_MODEL_ROOT="$CODE_ROOT/eval_output/waymo_reconstruction_longtail_hier_v4_first15"
EVAL_OUTPUT_ROOT="$CODE_ROOT/eval_output/longtail_hier_v4_first15_metrics"
MANAGED_SCALE2_ROOT="$CODE_ROOT/eval_output/managed_scale2_final_ckpts"

make_timestamp() {
  date +"%Y%m%d_%H%M%S"
}

ensure_run_tag_for_training() {
  if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG=$(make_timestamp)
  fi
}

build_model_path() {
  local model_root="$1"
  printf '%s/%s_%s\n' "$model_root" "$SCENE_ID" "$RUN_TAG"
}

resolve_latest_model_path() {
  local model_root="$1"
  local latest_path
  latest_path=$(find "$model_root" -maxdepth 1 -mindepth 1 -type d -name "${SCENE_ID}_*" | sort | tail -n 1)
  if [[ -z "$latest_path" ]]; then
    echo "No existing run found under $model_root for $SCENE_ID" >&2
    exit 1
  fi
  printf '%s\n' "$latest_path"
}

resolve_run_tag_from_model_path() {
  local model_path="$1"
  local base_name
  base_name=$(basename "$model_path")
  printf '%s\n' "${base_name#${SCENE_ID}_}"
}

copy_if_exists() {
  local src_path="$1"
  local dest_dir="$2"
  if [[ -e "$src_path" ]]; then
    cp -r "$src_path" "$dest_dir/"
  fi
}

run_train() {
  ensure_run_tag_for_training
  local model_path
  model_path=$(build_model_path "$RECON_MODEL_ROOT")
  mkdir -p "$model_path"
  cd "$CODE_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python train_hierarchical.py \
    --config "$RECON_CONFIG" \
    source_path="$DATA_PATH" \
    model_path="$model_path" \
    start_frame=$START_FRAME \
    end_frame=$END_FRAME \
    save_scale2_final_checkpoint=true \
    scale2_final_checkpoint_root="$MANAGED_SCALE2_ROOT" \
    enable_long_tail_branch=true \
    cam_num=3
}

run_eval() {
  local model_path
  if [[ -n "$RUN_TAG" ]]; then
    model_path=$(build_model_path "$RECON_MODEL_ROOT")
  else
    model_path=$(resolve_latest_model_path "$RECON_MODEL_ROOT")
    RUN_TAG=$(resolve_run_tag_from_model_path "$model_path")
  fi
  cd "$CODE_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_longtail_regions_hier.py \
    --config "$RECON_CONFIG" \
    source_path="$DATA_PATH" \
    model_path="$model_path" \
    start_frame=$START_FRAME \
    end_frame=$END_FRAME \
    enable_long_tail_branch=true \
    cam_num=3

  local latest_iter
  latest_iter=$(find "$model_path" -maxdepth 1 -type f -name 'chkpnt*.pth' | sed 's/.*chkpnt\([0-9]\+\)\.pth/\1/' | sort -n | tail -n 1)
  if [[ -z "$latest_iter" ]]; then
    echo "No checkpoint found in $model_path" >&2
    exit 1
  fi

  local dest_dir="$EVAL_OUTPUT_ROOT/reconstruction_${SCENE_ID}_${RUN_TAG}_iter${latest_iter}"
  mkdir -p "$dest_dir"
  copy_if_exists "$model_path/eval/region_metrics_${latest_iter}.json" "$dest_dir"
  copy_if_exists "$model_path/eval/region_metrics_${latest_iter}.md" "$dest_dir"
  copy_if_exists "$model_path/eval/test_${latest_iter}_region_metrics" "$dest_dir"
  copy_if_exists "$model_path/eval/train_${latest_iter}_region_metrics" "$dest_dir"
  copy_if_exists "$model_path/eval/test_${latest_iter}_videos" "$dest_dir"
  copy_if_exists "$model_path/eval/train_${latest_iter}_videos" "$dest_dir"
  copy_if_exists "$model_path/eval/test_${latest_iter}_render_only" "$dest_dir"
  copy_if_exists "$model_path/eval/train_${latest_iter}_render_only" "$dest_dir"
}

case "$TASK" in
  recon_full)
    run_train
    run_eval
    ;;
  train_recon)
    run_train
    ;;
  eval_recon)
    run_eval
    ;;
  *)
    echo "Usage: bash $0 [GPU_ID] [recon_full|train_recon|eval_recon] [RUN_TAG]" >&2
    exit 1
    ;;
esac
