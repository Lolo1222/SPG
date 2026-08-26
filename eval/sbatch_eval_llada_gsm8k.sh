#!/bin/bash
#SBATCH --job-name=eval_llada
#SBATCH --output=logs_eval/eval_llada_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --account=storygen
#SBATCH --qos=storygen_high

# Activate conda environment
# Initialize conda (adjust path if your conda is installed elsewhere)
# source /home/jwliu/miniconda3/etc/profile.d/conda.sh
# conda activate spg
cd /home/jwliu/dlm/SPG/eval

# Print environment info for debugging
echo "Python version: $(python --version)"
echo "PyTorch installation check:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || echo "PyTorch not found"
which torchrun || echo "torchrun not found in PATH"

# Configuration variables
# GPU_IDS will be automatically set by SLURM, but we'll use all available GPUs
GPU_IDS=(6)

# Generate a random port number between 10000 and 65535
MASTER_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $MASTER_PORT"

# Arrays of tasks and generation lengths
TASKS=("gsm8k")
# TASKS=("gsm8k" "math" "countdown" "sudoku")
GEN_LENGTHS=(256)
# GEN_LENGTHS=(128 256 512)
SAVE_DIR=/home/jwliu/dlm/SPG/save_dir

# Use SLURM allocated GPUs
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  # If SLURM has set CUDA_VISIBLE_DEVICES, use those GPUs
  IFS=',' read -ra SLURM_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  GPU_IDS=("${SLURM_GPUS[@]}")
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "Using GPUs: $GPU_LIST (nproc_per_node=$NUM_GPUS)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do
    # Set batch size based on generation length
    if [ "$gen_length" -eq 512 ]; then
      batch_size=4
    else
      batch_size=8
    fi
    
    echo "Running evaluation on $task with gen_length=$gen_length, batch_size=$batch_size"
    
    CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
      --nproc_per_node $NUM_GPUS \
      --master_port $MASTER_PORT \
      eval.py \
      --dataset $task \
      --batch_size $batch_size \
      --gen_length $gen_length \
      --output_dir "${SAVE_DIR}/new_eval_results/gsm8k_token_spg_mix_num_t3_20260523_091307" \
      --model_path "/home/jwliu/dlm/SPG/save_dir/token_spg_mix_num_t3_20260523_091307/checkpoint-6000"

      # --model_path "${SAVE_DIR}/hf_models/LLaDA-8B-Instruct/"
    # CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    #   --nproc_per_node $NUM_GPUS \
    #   --master_port $MASTER_PORT \
    #   eval.py \
    #   --dataset $task \
    #   --batch_size $batch_size \
    #   --gen_length $gen_length \
    #   --output_dir "${SAVE_DIR}/eval_results/eval_results_${task}_llada" \
    #   --model_path "${SAVE_DIR}/hf_models/LLaDA-8B-Instruct/" 
  done
done

# for gen_length in "${GEN_LENGTHS[@]}"; do
#   # Set batch size based on generation length
#   if [ "$gen_length" -eq 512 ]; then
#     batch_size=4
#   else
#     batch_size=8
#   fi
  
#   echo "Running evaluation on sudoku with gen_length=$gen_length, batch_size=$batch_size"
  
#   CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
#     --nproc_per_node $NUM_GPUS \
#     --master_port $MASTER_PORT \
#     eval.py \
#     --dataset "sudoku" \
#     --batch_size $batch_size \
#     --gen_length $gen_length \
#     --few_shot 3 \
#     --output_dir "${SAVE_DIR}/eval_results/eval_results_sudoku_new_3shot_llada" \
#     --model_path "${SAVE_DIR}/hf_models/LLaDA-8B-Instruct/" 
# done


echo "All evaluations completed!" 