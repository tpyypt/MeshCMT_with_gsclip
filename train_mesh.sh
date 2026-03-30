#!/bin/bash

set -e

DEVICE=${DEVICE:-0}
DATA_ROOT=${DATA_ROOT:-/data/tpy/datasets/Manifold40}
CACHE_ROOT=${CACHE_ROOT:-./.cache/manifold40}
META_PATH=${META_PATH:-}
EXP_DIR=${EXP_DIR:-./my_exp/mesh_cls_run}
EPOCH_GEOM=${EPOCH_GEOM:-30}
EPOCH1=${EPOCH1:-15}
EPOCH2=${EPOCH2:-15}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
RENDER_BACKEND=${RENDER_BACKEND:-pyrender_egl}
MESH_MASK_RATIO=${MESH_MASK_RATIO:-0.0}
STAGE1_RESUME=${STAGE1_RESUME:-}
STAGE1_GEOM_RESUME=${STAGE1_GEOM_RESUME:-}
STAGE1_FREEZE_MESH_ENCODER=${STAGE1_FREEZE_MESH_ENCODER:-1}
STAGE1_TUNE_MESH_GLOBAL_PROJ_ONLY=${STAGE1_TUNE_MESH_GLOBAL_PROJ_ONLY:-1}
STAGE1_FREEZE_GEOMETRY_CLASSIFIER=${STAGE1_FREEZE_GEOMETRY_CLASSIFIER:-1}
STAGE2_USE_DEPTH_BRANCH=${STAGE2_USE_DEPTH_BRANCH:-1}

echo "[train_mesh] DEVICE=${DEVICE}"
echo "[train_mesh] DATA_ROOT=${DATA_ROOT}"
echo "[train_mesh] CACHE_ROOT=${CACHE_ROOT}"
echo "[train_mesh] EXP_DIR=${EXP_DIR}"
echo "[train_mesh] EPOCH_GEOM=${EPOCH_GEOM}"
echo "[train_mesh] EPOCH1=${EPOCH1}"
echo "[train_mesh] EPOCH2=${EPOCH2}"
echo "[train_mesh] BATCH_SIZE=${BATCH_SIZE}"
echo "[train_mesh] NUM_WORKERS=${NUM_WORKERS}"
echo "[train_mesh] RENDER_BACKEND=${RENDER_BACKEND}"
echo "[train_mesh] MESH_MASK_RATIO=${MESH_MASK_RATIO}"
echo "[train_mesh] STAGE1_GEOM_RESUME=${STAGE1_GEOM_RESUME}"
echo "[train_mesh] STAGE1_RESUME=${STAGE1_RESUME}"
echo "[train_mesh] STAGE1_FREEZE_MESH_ENCODER=${STAGE1_FREEZE_MESH_ENCODER}"
echo "[train_mesh] STAGE1_TUNE_MESH_GLOBAL_PROJ_ONLY=${STAGE1_TUNE_MESH_GLOBAL_PROJ_ONLY}"
echo "[train_mesh] STAGE1_FREEZE_GEOMETRY_CLASSIFIER=${STAGE1_FREEZE_GEOMETRY_CLASSIFIER}"
echo "[train_mesh] STAGE2_USE_DEPTH_BRANCH=${STAGE2_USE_DEPTH_BRANCH}"

if [ -n "${STAGE1_GEOM_RESUME}" ] && [ ! -f "${STAGE1_GEOM_RESUME}" ]; then
  echo "[train_mesh] error: STAGE1_GEOM_RESUME checkpoint not found: ${STAGE1_GEOM_RESUME}" >&2
  exit 1
fi
if [ -n "${STAGE1_RESUME}" ] && [ ! -f "${STAGE1_RESUME}" ]; then
  echo "[train_mesh] error: STAGE1_RESUME checkpoint not found: ${STAGE1_RESUME}" >&2
  exit 1
fi

STAGE1_GEOM_ARGS=(
  --data_root "${DATA_ROOT}"
  --save_path "${EXP_DIR}/stage1_geom"
  --cache_root "${CACHE_ROOT}"
  --image_size 336
  --features_list 24
  --epochs "${EPOCH_GEOM}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --learning_rate 0.001
  --weight_decay 0.0001
  --label_smoothing 0.05
  --mesh_mask_ratio "${MESH_MASK_RATIO}"
  --render_backend "${RENDER_BACKEND}"
  --disable_render_cache_generation
)
if [ -n "${STAGE1_GEOM_RESUME}" ]; then
  STAGE1_GEOM_ARGS+=(--resume_checkpoint_path "${STAGE1_GEOM_RESUME}")
fi

CUDA_VISIBLE_DEVICES=${DEVICE} python train_mesh_stage1_geom.py \
  "${STAGE1_GEOM_ARGS[@]}"

STAGE1_ARGS=(
  --data_root "${DATA_ROOT}"
  --save_path "${EXP_DIR}/stage1"
  --cache_root "${CACHE_ROOT}"
  --image_size 336
  --features_list 24
  --epochs "${EPOCH1}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --learning_rate 0.002
  --prompt_learning_rate 0.0005
  --weight_decay 0.0001
  --label_smoothing 0.1
  --mesh_mask_ratio "${MESH_MASK_RATIO}"
  --init_geometry_checkpoint_path "${EXP_DIR}/stage1_geom/epoch_${EPOCH_GEOM}.pth"
  --render_backend "${RENDER_BACKEND}"
  --disable_render_cache_generation
)
if [ "${STAGE1_FREEZE_MESH_ENCODER}" = "1" ]; then
  STAGE1_ARGS+=(--freeze_mesh_encoder)
else
  STAGE1_ARGS+=(--no-freeze_mesh_encoder)
fi
if [ "${STAGE1_TUNE_MESH_GLOBAL_PROJ_ONLY}" = "1" ]; then
  STAGE1_ARGS+=(--tune_mesh_global_proj_only)
else
  STAGE1_ARGS+=(--no-tune_mesh_global_proj_only)
fi
if [ "${STAGE1_FREEZE_GEOMETRY_CLASSIFIER}" = "1" ]; then
  STAGE1_ARGS+=(--freeze_geometry_classifier)
else
  STAGE1_ARGS+=(--no-freeze_geometry_classifier)
fi
if [ -n "${STAGE1_RESUME}" ]; then
  STAGE1_ARGS+=(--resume_checkpoint_path "${STAGE1_RESUME}")
fi

CUDA_VISIBLE_DEVICES=${DEVICE} python train_mesh_stage1.py \
  "${STAGE1_ARGS[@]}"

python data_preprocess/render_mesh_multiview.py \
  --data_root "${DATA_ROOT}" \
  --cache_root "${CACHE_ROOT}" \
  --image_size 336 \
  --num_views 9 \
  --render_backend "${RENDER_BACKEND}"

if [ "${STAGE2_USE_DEPTH_BRANCH}" = "1" ]; then
  STAGE2_DEPTH_ARG=--use_depth_branch
else
  STAGE2_DEPTH_ARG=--no-use_depth_branch
fi

CUDA_VISIBLE_DEVICES=${DEVICE} python train_mesh_stage2.py \
  --data_root "${DATA_ROOT}" \
  --save_path "${EXP_DIR}/stage2" \
  --stage1_checkpoint_path "${EXP_DIR}/stage1/epoch_${EPOCH1}.pth" \
  --cache_root "${CACHE_ROOT}" \
  --image_size 336 \
  --features_list 24 \
  --epochs "${EPOCH2}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --learning_rate 0.001 \
  --direct_loss_weight 0.3 \
  --geometry_distill_weight 0.2 \
  --mesh_mask_ratio "${MESH_MASK_RATIO}" \
  --text_loss_weight 1.0 \
  --combined_loss_weight 0.0 \
  --text_logit_weight 0.3 \
  --render_backend "${RENDER_BACKEND}" \
  --disable_render_cache_generation \
  "${STAGE2_DEPTH_ARG}"
CUDA_VISIBLE_DEVICES=${DEVICE} python test_mesh_cls.py \
  --data_root "${DATA_ROOT}" \
  --save_path "${EXP_DIR}/eval" \
  --stage2_checkpoint_path "${EXP_DIR}/stage2/epoch_${EPOCH2}.pth" \
  --cache_root "${CACHE_ROOT}" \
  --image_size 336 \
  --features_list 24 \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --mesh_mask_ratio "${MESH_MASK_RATIO}" \
  --text_logit_weight 0.3 \
  --render_backend "${RENDER_BACKEND}" \
  --disable_render_cache_generation \
  "${STAGE2_DEPTH_ARG}"
