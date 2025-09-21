#! /bin/bash

module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse

run_id=$1

########################################################################
cp ${BASE_DIR}/Dataset/QG768.h5 /dev/shm/ &

wait

########################################################################
bash ${BASE_DIR}/Supervised/Experiments/Scripts/bash/AggregationTransformer768.sh ${run_id} &

wait