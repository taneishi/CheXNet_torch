#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N chexnet
#PBS -j oe
#PBS -o output.log

if [ ${PBS_O_WORKDIR} ]; then
    cd ${PBS_O_WORKDIR}
fi

CPUS=2
CORES=24
TOTAL_CORES=$((${CPUS}*${CORES}))

echo "CPUS=${CPUS} CORES=${CORES} TOTAL_CORES=${TOTAL_CORES}"
export OMP_NUM_THREADS=${TOTAL_CORES}
export KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

python infer.py --mode torch --batch_size 10 --test_image_list labels/test_list.txt
python infer.py --mode fp32 --batch_size 10 --test_image_list labels/test_list.txt
python infer.py --mode int8 --batch_size 10 --test_image_list labels/test_list.txt
