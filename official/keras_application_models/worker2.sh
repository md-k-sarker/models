#!/bin/bash
unset http_proxy
unset https_proxy
rm -rf /tmp/zhouhaiy
export OMP_NUM_THREADS=56
export KMP_HW_SUBSET=28c,1T
export KMP_AFFINITY=compact,granularity=fine
export KMP_BLOCKTIME=1
export PYTHONPATH=${PYTHONPATH}:/home/zhouhaiy/tensorflow/train-model/models
export TF_CONFIG='{
    "cluster": {
          "worker": ["192.168.20.57:5000", "192.168.20.58:5001"]
          },
    "task": {"type": "worker", "index": 1}
}'
python official/keras_application_models/benchmark_main.py --model densenet121
