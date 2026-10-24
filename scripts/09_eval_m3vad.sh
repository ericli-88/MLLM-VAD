#!/bin/bash
set -euo pipefail

dataset_dir="YOUR_DATASET_PATH"
experiment_dir="$dataset_dir/m3vad/experiment_e"

python -m src.evaluate_m3vad \
    --root_path "$dataset_dir/frames" \
    --annotationfile_path "$dataset_dir/annotations/test.txt" \
    --temporal_annotation_file "$dataset_dir/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt" \
    --frame_scores_dir "$experiment_dir/frame_scores" \
    --output_path "$experiment_dir/metrics.json" \
    --normal_label 0
