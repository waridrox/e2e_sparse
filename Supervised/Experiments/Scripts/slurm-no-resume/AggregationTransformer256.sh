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

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG256.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_256_S \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_256_S \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_256_S_${run_id} &

########################################################################

CUDA_VISIBLE_DEVICES=1 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG256.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_256_M \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_256_M \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_256_M_${run_id} &

########################################################################

CUDA_VISIBLE_DEVICES=2 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG256.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_256_L \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_256_L \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_256_L_${run_id} &


wait