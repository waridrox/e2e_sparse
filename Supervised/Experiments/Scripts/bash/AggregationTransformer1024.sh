#! /bin/bash

module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse

run_id=$1

########################################################################
cp ${BASE_DIR}/Dataset/QG1024.h5 /dev/shm/ &

wait

########################################################################

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_1024_S \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_1024_S \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_1024_S_${run_id} \
  --NAccumSteps=8 &

########################################################################

CUDA_VISIBLE_DEVICES=1 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_1024_M \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_1024_M \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_1024_M_${run_id}  \
  --NAccumSteps=8 &

########################################################################

CUDA_VISIBLE_DEVICES=2 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_1024_L \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_1024_L \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_1024_L_${run_id} \
  --NAccumSteps=8 &
########################################################################

CUDA_VISIBLE_DEVICES=3 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=/dev/shm/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=Transformer_PC_1024_H \
  --UseWandb=True \
  --wandb_project=AggregationTransformer \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=Transformer_PC_1024_H \
  --wandb_key=$wandb_key \
  --Checkpoint_dir=${BASE_DIR}/Supervised/Experiments/Checkpoints/Transformer_PC_1024_H_${run_id} \
  --NAccumSteps=8 &


wait
