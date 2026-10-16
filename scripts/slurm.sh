#!/bin/bash -x
#SBATCH --account=e-ext-2025e01-128
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:4
#SBATCH --chdir=.
#SBATCH --partition=booster
#SBATCH --output=logs/weathergen-%x.%j.out
#SBATCH --error=logs/weathergen-%x.%j.err

echo "shell $SHELL"
if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
fi

. "$HOME/.bashrc"
echo "LMOD_ROOT $LMOD_ROOT"
# Does not get loaded automatically. Unsure why.
. $LMOD_ROOT/lmod/init/bash
# Load basic modules from software stack
module --force purge
module load Stages/2025

module load GCC/13.3.0
module load GCCcore/.13.3.0

ml git/2.45.1
ml Python/3.12.3
if [ "${WEATHERGEN_NSYS_PROFILING:-0}" = "1" ]; then
    ml Nsight-Systems/2024.7.1
fi


# set some environment variables to ensure that GPU-devices are used properly
export UCX_TLS="^cma"
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_4:1,mlx5_5:1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0,1,2,3

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
if [ "$SYSTEMNAME" = juwelsbooster ] \
       || [ "$SYSTEMNAME" = juwels ] \
       || [ "$SYSTEMNAME" = jurecadc ] \
       || [ "$SYSTEMNAME" = jusuf ]; then
    # Allow communication over InfiniBand cells on JSC machines.
    MASTER_ADDR="$MASTER_ADDR"i
fi
echo "MASTER_ADDR: $MASTER_ADDR"

export NCCL_DEBUG=INFO
echo "nccl_debug: $NCCL_DEBUG"

echo "Starting job."
date
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if WEATHERGEN_HOME is set
if [ -z "$WEATHERGEN_HOME" ]; then
    echo "WEATHERGEN_HOME is not set"
    exit 1
fi


# Check if WEATHERGEN_STAGE is set
# if [ -z "$WEATHERGEN_STAGE" ]; then
#     echo "WEATHERGEN_STAGE is not set"
#     exit 1
# fi

WEATHERGEN_PRIVATE_REPO_PATH="${WEATHERGEN_PRIVATE_REPO_PATH:-$(dirname "$WEATHERGEN_HOME")/WeatherGenerator-private}"
source "${WEATHERGEN_PRIVATE_REPO_PATH}/hpc/nsys_profiling.sh"

echo "hello jupiter! the environment is setup"

export WORLD_SIZE=1 # just a gaurd against an accidental ntasks>1 request in slurm options
./scripts/actions.sh integration-test