#!/bin/sh
#export CUDA_VISIBLE_DEVICES=0
#export LD_PRELOAD="/usr/lib/libtcmalloc.so"
NAME=${1?Error: no snapshot given}
checkpoint="./model.ckpt-"$NAME  # 32000"
# echo $checkpoint
save_path=${2}  # "./train_gradients/"
test_dataset=${4}
test_image_root="/home/chiragagarwall12/imagenet/train/"
output_file="./eval-0.pkl"
class_ind=${3}
nohup python train_get_gradients.py --checkpoint $checkpoint --test_dataset $test_dataset --test_image_root $test_image_root --output_file $output_file --batch_size 1 --test_iter 1300000 --ngroups1 1 --ngroups2 1 --gpu_fraction 0.0 --display 10 --class_ind $class_ind --save_path $save_path >/dev/null 2>&1 & 
