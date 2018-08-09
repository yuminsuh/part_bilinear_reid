#!/usr/bin/env bash

# Set paths
source paths
export MARKET1501_DATA_ROOT=$MARKET1501_DATA_ROOT
export MARKET1501_TRAIN_LIST=$MARKET1501_TRAIN_LIST
export INCEPTION_V1_PRETRAINED=$INCEPTION_V1_PRETRAINED
export CPM_PRETRAINED=$CPM_PRETRAINED

source $CONFIGURE_PATH

# Make log directory
if [ -d $LOG_DIR ]; then
    echo "Same experiment already exists. Change the exp name and retry!"
    exit
else
    mkdir -p $LOG_DIR
    cp $CONFIGURE_PATH "$LOG_DIR/args"
fi

# Parameters
STR_PARAM="-d $DATASET -b $BATCH_SIZE -j 4 -a $ARCH --logs-dir $LOG_DIR --margin 0.2 --features $FEATURES --width $WIDTH --height $HEIGHT --crop-height $CROP_HEIGHT --crop-width $CROP_WIDTH --lr $LR --epochs $EPOCHS --dilation $DILATION --weight-decay $WEIGHT_DECAY"

# Run!
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON train.py $STR_PARAM
