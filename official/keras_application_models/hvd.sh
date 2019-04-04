#!/bin/bash
rm -rf /tmp/zhouhaiy
source /opt/intel/compilers_and_libraries_2018/linux/mpi/intel64/bin/mpivars.sh
export OMP_NUM_THREADS=56
export KMP_HW_SUBSET=28c,1T
export KMP_AFFINITY=compact,granularity=fine
export KMP_BLOCKTIME=1
export PYTHONPATH=${PYTHONPATH}:/home/zhouhaiy/tensorflow/train-model/models

mpirun -n 2 -machinefile ./mpd.hosts -ppn 1 -genv  OMP_NUM_THREADS=56 -genv KMP_HW_SUBSET=28c,1T -genv KMP_AFFINITY=compact,granularity=fine -genv KMP_BLOCKTIME=1 python official/keras_application_models/benchmark_main.py --model densenet121



