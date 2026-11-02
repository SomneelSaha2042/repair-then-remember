#!/bin/bash
#SBATCH --job-name=repair-memory
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=repair-memory-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "$HOME/.venvs/alphagen/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); print((torch.ones(4, device="cuda") * 2).sum().item())'
python -m scripts.compare_repairs "$@"
