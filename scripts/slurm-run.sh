#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p gpu
#SBATCH -t 01:30:00
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=Project_2001710
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOBID.out logs/latest.out
ln -s $SLURM_JOBID.err logs/latest.err

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 SCRIPT [ARG[...]]" >&2
    exit 1
fi

script=$1
shift

module purge
module load tensorflow
source $HOME/venv/keras-bert/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID ($script): $(date)"

srun "$script" "$@"

echo "END $SLURM_JOBID ($script): $(date)"

seff $SLURM_JOBID
