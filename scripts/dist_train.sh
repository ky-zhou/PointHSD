#!/usr/bin/env bash

set -x
NGPUS=$1
PORT=$2



# python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} main17_pcn_ddp.py --launcher pytorch ${PY_ARGS}

# python -m torch.distributed.launch --nproc_per_node=${NGPUS}  --master_port=${PORT} main17_ddp.py
python -m torch.distributed.launch --nproc_per_node=${NGPUS}  --master_port=${PORT} main21_ddp.py