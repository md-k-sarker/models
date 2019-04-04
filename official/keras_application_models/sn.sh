#!/bin/bash
unset http_proxy
unset https_proxy
export OMP_NUM_THREADS=56
export KMP_HW_SUBSET=28c,1T
export KMP_AFFINITY=compact,granularity=fine
export KMP_BLOCKTIME=1
rm -rf /tmp/zhouhaiy
export PYTHONPATH=${PYTHONPATH}:/home/zhouhaiy/tensorflow/train-model/models
python official/keras_application_models/benchmark_main.py --model densenet121
