#!/usr/bin/env bash
set -euo pipefail

ROOT="/ssd2/wenyan/CarTwin/longtail/PVG"
SCENE_ID="0c1afba4-e796-43c6-ba36-6a225b6f2968"
DATA_PATH="$ROOT/data_longtail/$SCENE_ID"
GPU_ID="${1:-6}"
TASK="${2:-all}"
RUN_TAG="${3:-}"
START_FRAME=0
END_FRAME=48

RECON_CONFIG="configs/waymo_reconstruction_longtail_tuned.yaml"
NVS_CONFIG="configs/waymo_nvs_longtail_tuned.yaml"

RECON_MODEL_ROOT="$ROOT/eval_output/waymo_reconstruction_longtail_tuned_first49"
NVS_MODEL_ROOT="$ROOT/eval_output/waymo_nvs_longtail_tuned_first49"
EVAL_OUTPUT_ROOT="$ROOT/eval_output/longtail_tuned_region_metrics_first49"

make_timestamp() {
  date +"%Y%m%d_%H%M%S"
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

ensure_run_tag_for_training() {
  if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG=$(make_timestamp)
  fi
}

build_model_path() {
  local model_root="$1"
  printf '%s/%s_%s\n' "$model_root" "$SCENE_ID" "$RUN_TAG"
}

copy_if_exists() {
  local src_path="$1"
  local dest_dir="$2"
  if [[ -e "$src_path" ]]; then
    cp -r "$src_path" "$dest_dir/"
  fi
}

run_train() {
  local config_path="$1"
  local model_root="$2"
  ensure_run_tag_for_training
  local model_path
  model_path=$(build_model_path "$model_root")
  mkdir -p "$model_path"
  cd "$ROOT"
  echo "Training output: $model_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    --config "$config_path" \
    source_path="$DATA_PATH" \
    model_path="$model_path" \
    start_frame=$START_FRAME \
    end_frame=$END_FRAME \
    enable_long_tail_branch=true \
    cam_num=3
}

run_eval() {
  local config_path="$1"
  local model_root="$2"
  local tag="$3"
  local model_path
  if [[ -n "$RUN_TAG" ]]; then
    model_path=$(build_model_path "$model_root")
  else
    model_path=$(resolve_latest_model_path "$model_root")
    RUN_TAG=$(resolve_run_tag_from_model_path "$model_path")
  fi
  cd "$ROOT"
  echo "Evaluating model: $model_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python evaluate_longtail_regions.py \
    --config "$config_path" \
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

  local dest_dir="$EVAL_OUTPUT_ROOT/${tag}_${SCENE_ID}_${RUN_TAG}_iter${latest_iter}"
  mkdir -p "$dest_dir"
  copy_if_exists "$model_path/eval/region_metrics_${latest_iter}.json" "$dest_dir"
  copy_if_exists "$model_path/eval/region_metrics_${latest_iter}.md" "$dest_dir"
  copy_if_exists "$model_path/eval/test_${latest_iter}_region_metrics" "$dest_dir"
  copy_if_exists "$model_path/eval/train_${latest_iter}_region_metrics" "$dest_dir"
  echo "Saved $tag evaluation to $dest_dir"
}

case "$TASK" in
  train_recon)
    run_train "$RECON_CONFIG" "$RECON_MODEL_ROOT"
    ;;
  train_nvs)
    run_train "$NVS_CONFIG" "$NVS_MODEL_ROOT"
    ;;
  eval_recon)
    run_eval "$RECON_CONFIG" "$RECON_MODEL_ROOT" "reconstruction"
    ;;
  eval_nvs)
    run_eval "$NVS_CONFIG" "$NVS_MODEL_ROOT" "nvs"
    ;;
  all)
    run_train "$RECON_CONFIG" "$RECON_MODEL_ROOT"
    run_train "$NVS_CONFIG" "$NVS_MODEL_ROOT"
    run_eval "$RECON_CONFIG" "$RECON_MODEL_ROOT" "reconstruction"
    run_eval "$NVS_CONFIG" "$NVS_MODEL_ROOT" "nvs"
    ;;
  *)
    echo "Usage: bash $0 [GPU_ID] [train_recon|train_nvs|eval_recon|eval_nvs|all] [RUN_TAG]" >&2
    exit 1
    ;;
esac
