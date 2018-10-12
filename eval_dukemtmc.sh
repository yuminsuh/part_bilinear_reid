#!/usr/bin/env bash

# Download pretrained network weights
if ! [ -d "logs/dukemtmc/d2_b250" ]; then
    mkdir -p "logs/dukemtmc/d2_b250"
fi
if ! [ -f "logs/dukemtmc/d2_b250/args.json" ]; then
    wget "https://www.dropbox.com/s/tzgg3xkot4aosa2/args.json?dl=0" -O "logs/dukemtmc/d2_b250/args.json"
fi
if ! [ -f "logs/dukemtmc/d2_b250/epoch_750.pth.tar" ]; then
    wget "https://www.dropbox.com/s/50snw87h30yey93/epoch_750.pth.tar?dl=0" -O "logs/dukemtmc/d2_b250/epoch_750.pth.tar"
fi

python test.py -d 'dukemtmc' -e 'd2_b250' --epoch 750 --batchsize 50 --gpus '0'
