#!/usr/bin/env bash

# Download pretrained network weights
if ! [ -d "logs/mars/d2_b250" ]; then
    mkdir -p "logs/mars/d2_b250"
fi
if ! [ -f "logs/mars/d2_b250/args.json" ]; then
    wget "https://www.dropbox.com/s/b1zloaz6avh5mre/args.json?dl=0" -O "logs/mars/d2_b250/args.json"
fi
if ! [ -f "logs/mars/d2_b250/epoch_750.pth.tar" ]; then
    wget "https://www.dropbox.com/s/ef3ueupgr7msk41/epoch_750.pth.tar?dl=0" -O "logs/mars/d2_b250/epoch_750.pth.tar"
    
fi

python test.py
