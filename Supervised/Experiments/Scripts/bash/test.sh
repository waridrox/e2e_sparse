#! /bin/bash

module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse


########################################################################

CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_1024_S \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_1024_S \
  --wandb_key=$wandb_key &


  CUDA_VISIBLE_DEVICES=0 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG1024.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_1024_M \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_1024_M \
  --wandb_key=$wandb_key &
########################################################################

CUDA_VISIBLE_DEVICES=1 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG768.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_768_S \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_768_S \
  --wandb_key=$wandb_key &

  CUDA_VISIBLE_DEVICES=1 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG768.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_768_M \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_768_M \
  --wandb_key=$wandb_key &

########################################################################

CUDA_VISIBLE_DEVICES=2 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG512.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_512_S \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_512_S \
  --wandb_key=$wandb_key &


  CUDA_VISIBLE_DEVICES=1 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG512.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_512_M \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_512_M \
  --wandb_key=$wandb_key &

########################################################################

CUDA_VISIBLE_DEVICES=2 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG256.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_256_S \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_256_S \
  --wandb_key=$wandb_key &

  CUDA_VISIBLE_DEVICES=1 python3 ${BASE_DIR}/Supervised/trainer.py \
  --datapath=${BASE_DIR}/Dataset/QG256.h5 \
  --Nepochs=100 \
  --lr=1e-3 \
  --model_variant=ResNet_PC_256_M \
  --UseWandb=True \
  --wandb_project=SparseCNN \
  --wandb_entity=e2e_sparse \
  --wandb_run_name=ResNet_PC_256_M \
  --wandb_key=$wandb_key &

  wait