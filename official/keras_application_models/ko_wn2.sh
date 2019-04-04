#!/bin/bash

unset http_proxy
unset https_proxy
rm -rf /tmp/mdkamruz
export OMP_NUM_THREADS=48
export KMP_HW_SUBSET=28c,1T
export KMP_AFFINITY=compact,granularity=fine
export KMP_BLOCKTIME=1

# endeavour : "36.101.24.14:3001"
# folsom: 10.105.188.58
# ,"10.105.188.58:3002"
export TF_CONFIG='{
    "cluster": {
          "worker": ["10.105.188.58:3005","10.105.188.58:3007"]
          },
    "task": {"type": "worker", "index": 1}
}'

#export PYTHONPATH="$PYTHONPATH:/ec/fm/disks/aipg_lab_home_pool_03/mdkamruz/code/repos/models"

# start script for zhouhai/actual model
#python /ec/fm/disks/aipg_lab_home_pool_03/mdkamruz/code/repos/models/official/keras_application_models/benchmark_main_ds.py --model densenet121 2>&1 | tee ~/worker1_main_ds.txt
# --dist_strat --distribution_strategy one_device --model densenet121
# collective_allreduce 

# start script for test
python ~/code/test/distribution_overview.py 2>&1 | tee ~/worker1_keras_ov.txt