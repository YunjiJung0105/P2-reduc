#!/bin/bash

for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

gpu_num=$gpu
test_dir=$test_dir
data_dir=$data_dir
num_samples=20

pt_list=($(ls $test_dir | grep "ema"))
model_params="--attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --num_channels 128 --num_res_blocks 1 --num_head_channels 64 --resblock_updown True"
learning_params="--image_size 256 --learn_sigma True --noise_schedule linear --use_fp16 False --use_scale_shift_norm True"
sampling_params="--timestep_respacing ddim25 --use_ddim True"

for pt_file in ${pt_list[@]}
do
    CUDA_VISIBLE_DEVICES=$gpu_num python scripts/image_nn_sample.py --num_samples $num_samples --sample_dir samples/$output_dir --model_path $test_dir/$pt_file --data_dir $data_dir $model_params $learning_params $sampling_params
done