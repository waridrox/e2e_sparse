#! /bin/bash

module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse


python3 ${BASE_DIR}/Supervised/trainer.py --datapath=${BASE_DIR}/output.h5 --Nepochs=100 --lr=1e-3 --model_variant=ResNet_PC_768_S --UseWandb=True --wandb_project=mytest --wandb_entity=e2e_sparse --wandb_run_name=test --wandb_key=$wandb_key
