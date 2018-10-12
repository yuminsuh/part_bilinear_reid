#!/usr/bin/env bash

# Download pretrained network weights
if ! [ -d "logs/market1501/d2_b250" ]; then
    mkdir -p "logs/market1501/d2_b250"
fi
if ! [ -f "logs/market1501/d2_b250/args.json" ]; then
    wget "https://www.dropbox.com/s/e0cy15io2pwuzth/args.json?dl=0" -O "logs/market1501/d2_b250/args.json"
fi
if ! [ -f "logs/market1501/d2_b250/epoch_750.pth.tar" ]; then
    wget "https://www.dropbox.com/s/dugjdeav6iapjvt/epoch_750.pth.tar?dl=0" -O "logs/market1501/d2_b250/epoch_750.pth.tar"
    
fi

python test.py -d 'market1501' -e 'd2_b250' --epoch 750 --batchsize 50 --gpus '0'
