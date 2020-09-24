#!/bin/zsh                   
# echo "Hello  World"
echo "train protonet"
python train.py --method=protonet  --model=ResNet12 --dataset=CUB --n_shot=5 --n_query=8 --lr_anneal=const --start_epoch=499 --stop_epoch=600
echo "train am3"
python train.py --method=am3  --model=ResNet12 --dataset=CUB --n_shot=5 --n_query=8 --lr_anneal=const --start_epoch=499 --stop_epoch=600

# TO DO
# 
# 

