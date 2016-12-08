#!/usr/bin/env bash

export learningRate=0.1
export epoch_step="{100,150}"
export max_epoch=200
export learningRateDecay=0
export learningRateDecayRatio=0.1
export nesterov=true
export randomcrop_type=zero

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train.lua | tee $save/log.txt
