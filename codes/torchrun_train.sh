#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./conf/repre_learner10/train-streamconformer-bestRQ_0.6B-flex.yml "}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

CUR_DIR=`pwd`
PYTHON_TEMP=/home/wumenglin/repo-dev/CosyVoice-dev:/home/wumenglin/repo-dev/CosyVoice-dev/third_party/Matcha-TTS
export PYTHONPATH="${PYTHON_TEMP}:${PYTHONPATH}:${CUR_DIR}"
export TF_ENABLE_ONEDNN_OPTS=0
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
export MOSHI_DIR=/home/wumenglin/repo/moshi-dev/moshi

if [ ${NGPU} -gt 1 ]; then
    #PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    torchrun --nproc_per_node=${NGPU} \
    train.py -opt ${CONFIG_FILE} --launcher torchrun --gpus ${NGPU} $overrides
else
    #PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    python train.py -opt ${CONFIG_FILE} --launcher none --gpus ${NGPU} $overrides
fi
