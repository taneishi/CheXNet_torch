#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -N CheXNet_OpenVINO
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

python main.py --mode torch --batch_size 10
python main.py --mode fp32 --batch_size 10
python main.py --mode int8 --batch_size 10
