# Tail Miner V1

Strict single-frame long-tail miner for PVG multi-camera autonomous driving scenes.

Behavior:

1. Qwen3-VL proposes up to 3 text-only rare-object candidates for one camera frame using context `t-2 ... t+2`, with at most one candidate per label.
2. Local SAM3 uses each candidate text prompt to generate first-round masks and selects one primary core mask.
3. A local ROI is built from the primary core box with `2.5x` expansion and a `200 px` center allowance.
4. DINOv3 ViT-H/16 Plus dense patch features are extracted only inside that ROI.
5. High-similarity clusters become second-round SAM3 prompts.
6. The candidate result is one final merged `0/255` binary mask.
7. Only final per-instance masks are saved.

What this module does not do:

- no fallback proposal path
- no fallback segmentation path
- no VLM bounding-box proposal path
- no tracking
- no cross-camera association
- no 4D reconstruction
- no debug overlays, manifests, or intermediate saved masks

## Required models

- Qwen3-VL from `/ssd2/wenyan/CarTwin/longtail/Qwen3-VL-8B-Thinking`
- Local SAM3 from `/ssd2/wenyan/CarTwin/longtail/sam3`
- DINOv3 torch.hub repo `facebookresearch/dinov3`
- DINOv3 entrypoint `dinov3_vith16plus`
- DINOv3 local weights `/ssd2/wenyan/CarTwin/longtail/PVG/tail_miner/dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`

The code fails fast if any of these cannot be loaded.

For DINOv3, place the Meta checkpoint file at
`/ssd2/wenyan/CarTwin/longtail/PVG/tail_miner/dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth`.
The code loads that local `.pth` through the official `facebookresearch/dinov3` torch.hub entrypoint `dinov3_vith16plus`.

## Required Python packages

At minimum, this pipeline expects a recent `transformers` build for Qwen3-VL, plus `torch` / `torchvision` for the DINOv3 torch.hub backbone and the local SAM3 dependencies.

```bash
pip install transformers accelerate sentencepiece
```

## Run

```bash
python -m tail_miner \
  --scene-root /ssd2/wenyan/CarTwin/longtail/PVG/data_longtail/0c1afba4-e796-43c6-ba36-6a225b6f2968 \
  --output-root /ssd2/wenyan/CarTwin/longtail/PVG/tail_miner/output_v1 \
  --cameras front left right \
  --frame-start 0 \
  --frame-end 20 \
  --frame-stride 1 \
  --vlm-context-size 5 \
  --max-candidates-per-frame 3
```

For your current setup:

- example scene: `/ssd2/wenyan/CarTwin/longtail/PVG/data_longtail/0c1afba4-e796-43c6-ba36-6a225b6f2968`
- target workload: about 50 frames
- GPU: RTX A6000 49GB

Recommended command for a 50-frame run:

```bash
cd /ssd2/wenyan/CarTwin/longtail/PVG

python -m tail_miner \
  --scene-root /ssd2/wenyan/CarTwin/longtail/PVG/data_longtail/0c1afba4-e796-43c6-ba36-6a225b6f2968 \
  --output-root /ssd2/wenyan/CarTwin/longtail/PVG/tail_miner/output_v1 \
  --cameras front left right \
  --frame-start 0 \
  --frame-end 49 \
  --frame-stride 1 \
  --frame-cache-size 96 \
  --vlm-context-size 5 \
  --max-candidates-per-frame 3
```

Why this is better for 50 frames:

- `frame_end 49`: processes about 50 target frames
- `vlm-context-size 5`: VLM sees `t-2 ... t+2`, which is still compact
- `frame-cache-size 96`: avoids repeated PNG decoding across neighboring frames and cameras
- processing is still one target frame at a time, so memory pressure stays reasonable on a 49GB A6000

VLM temporal context:

- with `--vlm-context-size 5`, the VLM sees `t-2, t-1, t, t+1, t+2`
- context is single-camera only
- near the start or end of a clip, boundary frames are clamped
- VLM output is semantic only: `candidate_label`, `candidate_text_prompt`, `confidence`
- VLM is instructed to return at most one candidate per label (`bird`, `small_animal`, `unknown_small_object`)
- first-round segmentation comes from SAM3 text prompting, not from a VLM box

Output layout:

```text
output_root/original_longtail_masks/<scene_id>/<camera>/<frame_idx:06d>/mask_<instance_idx:02d>.png
output_root/dino_helped_longtail_masks/<scene_id>/<camera>/<frame_idx:06d>/mask_<instance_idx:02d>.png
```

Each saved mask:

- matches the original frame resolution
- is `uint8`
- uses `255` for foreground and `0` for background

Folder meaning:

- `original_longtail_masks`: the first-round SAM3 primary masks before DINO-guided refinement
- `dino_helped_longtail_masks`: the final refined masks after DINO-guided cluster discovery, second-round SAM3, and frame-level deduplication
