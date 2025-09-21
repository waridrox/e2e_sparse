#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4

export SLURM_CPU_BIND="cores"
srun ./gpus_for_tasks


#! /bin/bash

module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse

run_id=$1

########################################################################
cp ${BASE_DIR}/Dataset/QG256.h5 /dev/shm/ &

wait

########################################################################
bash ${BASE_DIR}/Supervised/Experiments/Scripts/bash/AggregationTransformer256.sh ${run_id} &

wait