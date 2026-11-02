#!/bin/bash
#SBATCH --job-name=repair-pilot
#SBATCH --array=0-2%1
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=repair-pilot-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$HOME/.venvs/alphagen/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
conditions=(harla repair memory)
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); print((torch.ones(4, device="cuda") * 2).sum().item())'
python -m scripts.compare_repairs --data data/qlib_bin --profile postrelease \
  --conditions "${conditions[$SLURM_ARRAY_TASK_ID]}" --seeds 0 --steps 51200 \
  --llm-updates 20 --output "out/gpu-pilot/${SLURM_ARRAY_JOB_ID}" "$@"
