#!/bin/bash
set -euo pipefail

dataset_dir="YOUR_DATASET_PATH"
experiment_config="configs/experiment_e_robust_memory.json"
llm_model="YOUR_INSTRUCTION_TUNED_MODEL"

python -m src.pipeline.m3vad_pipeline \
    --config "$experiment_config" \
    --root_path "$dataset_dir/frames" \
    --annotationfile_path "$dataset_dir/annotations/test.txt" \
    --captions_dir "$dataset_dir/captions/m3vad_blip2" \
    --video_dir "$dataset_dir/videos" \
    --output_dir "$dataset_dir/m3vad/experiment_e" \
    --llm_backend transformers \
    --llm_model "$llm_model"
