#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --job-name=AggregationTransformer768

timeout_handler()
{
    scontrol requeue ${SLURM_JOB_ID}
}


module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse

run_id=$1

########################################################################
cp ${BASE_DIR}/Dataset/QG768.h5 /dev/shm/ &

wait

########################################################################
bash ${BASE_DIR}/Supervised/Experiments/Scripts/bash/AggregationTransformer768.sh ${run_id} &

trap "timeout_handler" USR1

wait