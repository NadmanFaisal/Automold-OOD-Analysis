#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00             # Adjust time as needed
#SBATCH --account=naiss2026-4-688-gpu

# --- UPDATE THESE TWO PATHS ---
PROJECT_DIR="/nobackup/proj/disk/av-ood-benchmarking/personal/seasonal-weather-ood/Automold-OOD-Analysis"
CONTAINER="/nobackup/proj/disk/av-ood-benchmarking/personal/seasonal-weather-ood/Automold-OOD-Analysis/ood_evaluator.sif"

# 1. Define Master Variables
export OOD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export BASELINE_ID="20260502_220411"

WEATHERS=("Fog" "Snow")
SEVERITIES=("easy" "mid" "hard")

for CURRENT_WEATHER in "${WEATHERS[@]}"; do
  for CURRENT_SEVERITY in "${SEVERITIES[@]}"; do

    echo "========================================================="
    echo "STARTING FULL PIPELINE FOR: $CURRENT_WEATHER ($CURRENT_SEVERITY)"
    echo "========================================================="
    
    export OOD_WEATHER=$CURRENT_WEATHER
    export OOD_SEVERITY=$CURRENT_SEVERITY
    export OOD_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

    # ---------------------------------------------------------
    # Step 1: Run BEVFormer Evaluation (4 GPUs)
    # ---------------------------------------------------------
    echo "[$OOD_WEATHER $OOD_SEVERITY] Running Step 1: Model Evaluation..."
    apptainer exec --nv \
        --bind $PROJECT_DIR/data:/workspace/data \
        --bind $PROJECT_DIR/data/sets/nuscenes-automold:/workspace/data/nuscenes-automold \
        --bind $PROJECT_DIR/data/sets/nuscenes:/workspace/data/nuscenes \
        --bind $PROJECT_DIR/data/sets/nuscenes:/workspace/nuscenes:ro \
        --bind $PROJECT_DIR/checkpoints:/workspace/checkpoints \
        --bind $PROJECT_DIR/plots:/workspace/plots \
        --bind $PROJECT_DIR/core_models/BEVFormer/.dist_test:/workspace/core_models/BEVFormer/.dist_test:rw \
        --bind $PROJECT_DIR/evaluation_results:/workspace/test:rw \
        --pwd /workspace \
        --env PYTHONPATH="/workspace/core_models/BEVFormer" \
        --env OOD_WEATHER=$OOD_WEATHER,OOD_SEVERITY=$OOD_SEVERITY,OOD_TIMESTAMP=$OOD_TIMESTAMP \
        $CONTAINER \
        torchrun --nproc_per_node=4 /workspace/core_models/BEVFormer/tools/test.py \
            /workspace/core_models/BEVFormer/projects/configs/bevformer/bevformer_base.py \
            /workspace/checkpoints/bevformer/bevformer_r101_dcn_24ep.pth \
            --launcher pytorch \
            --eval bbox \
            --tmpdir /tmp/bevformer_eval

    
    echo "========================================================="
    echo "PIPELINE COMPLETELY FINISHED!"
    echo "========================================================="
  done
done

echo "========================================================="
echo "ALL WEATHERS AND SEVERITIES COMPLETELY FINISHED!"
echo "========================================================="
