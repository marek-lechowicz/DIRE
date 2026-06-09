#!/bin/bash
# Evaluate DIRE on the FakeFlickr dataset (Flickr30k test split).
#
# Single-GPU, MPI-free: DIRE maps are produced by guided-diffusion/
# compute_dire_single.py (no mpiexec required).  Uses the project venv (.venv).
#
# Override via env vars:
#   CKPT            - DIRE classifier checkpoint (.pth)
#   DIFFUSION_CKPT  - unconditional guided-diffusion checkpoint (256x256_diffusion_uncond.pt)
#   GPU            - CUDA device index to use
#   DBATCH         - diffusion batch size (VRAM: ~19GB at 24, ~13GB at 16)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$REPO_ROOT/.venv/bin/python}"

DATASET_ROOT="/home/marek/FakeFlickr/data/fake-flickr"
TEST_SPLIT="/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"

CKPT="${CKPT:-dire_models/lsun_adm.pth}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-/home/marek/Downloads/256x256_diffusion_uncond.pt}"
GPU="${GPU:-0}"
DBATCH="${DBATCH:-16}"

# DEBUG=1 ./test_fake_flickr.sh            -> 10 images / set
# DEBUG=1 DEBUG_SAMPLES=5 ./test_...        -> 5 images / set
DEBUG_FLAGS=()
if [ "${DEBUG:-0}" = "1" ]; then
    DEBUG_FLAGS+=(--debug --debug-samples "${DEBUG_SAMPLES:-10}")
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PY" test_fake_flickr.py \
    --dataset-root "$DATASET_ROOT" \
    --test-split "$TEST_SPLIT" \
    --ckpt "$CKPT" \
    --diffusion-ckpt "$DIFFUSION_CKPT" \
    --cuda-visible-devices "$GPU" \
    --diffusion-batch-size "$DBATCH" \
    --work-dir eval_work/fake_flickr \
    --results-csv eval_work/fake_flickr_dire.csv \
    "${DEBUG_FLAGS[@]}" \
    "$@"
