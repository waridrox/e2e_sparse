#! /bin/bash

module load tensorflow/2.12.0

BASE_DIR=${PSCRATCH}/e2e_sparse

if [ ! -d ${BASE_DIR}/Dataset ]; then
  mkdir -p ${BASE_DIR}/Dataset
fi

if [ -f ${BASE_DIR}/Dataset/QG1024.h5 ]; then
  echo "Deleting existing file: ${BASE_DIR}/Dataset/QG1024.h5"
  rm -rf ${BASE_DIR}/Dataset/QG1024.h5
fi

if [ -f ${BASE_DIR}/Dataset/QG768.h5 ]; then
  echo "Deleting existing file: ${BASE_DIR}/Dataset/QG768.h5"
  rm -rf ${BASE_DIR}/Dataset/QG768.h5
fi

if [ -f ${BASE_DIR}/Dataset/QG512.h5 ]; then
  echo "Deleting existing file: ${BASE_DIR}/Dataset/QG512.h5"
  rm -rf ${BASE_DIR}/Dataset/QG512.h5
fi

if [ -f ${BASE_DIR}/Dataset/QG256.h5 ]; then
  echo "Deleting existing file: ${BASE_DIR}/Dataset/QG256.h5"
  rm -rf ${BASE_DIR}/Dataset/QG256.h5
fi

python3 ToPointCloudForm.py --OutFile=${BASE_DIR}/Dataset/QG1024.h5 --MaxPoints=1024 &
python3 ToPointCloudForm.py --OutFile=${BASE_DIR}/Dataset/QG768.h5 --MaxPoints=768 &
python3 ToPointCloudForm.py --OutFile=${BASE_DIR}/Dataset/QG512.h5 --MaxPoints=512 &
python3 ToPointCloudForm.py --OutFile=${BASE_DIR}/Dataset/QG256.h5 --MaxPoints=256 &

wait