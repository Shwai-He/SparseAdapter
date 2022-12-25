#! /bin/bash

dataset_name=xsum
model_name_or_path=facebook/bart-large
num_train_epochs=10
weight_decay=0.1
learning_rate=1e-4
pruner=rand
bsz=48
device_ids="0 1 2 3 4 5 6 7"
DATE=`date +%Y%m%d`
attn_mode="adapter"
ffn_mode="adapter"
attn_bn=64
ffn_bn=64
attn_option=sequential
ffn_option=sequential
sparsity=1.
metric_for_best_model=rouge2
SAVE=./checkpoints/${model_name_or_path}/${TASK_NAME}/${DATE}/${pruner}_${sparsity}_${attn_mode}_${attn_option}_${attn_bn}_${ffn_mode}_${ffn_option}_${ffn_bn}
echo "${SAVE}"
mkdir -p ${SAVE}

nohup python ./run_summarization_sparse.py \
      --model_name_or_path ${model_name_or_path} \
      --dataset_name ${dataset_name} \
      --do_train --do_eval --overwrite_output_dir \
      --per_device_train_batch_size ${bsz} --per_device_eval_batch_size ${bsz} \
      --output_dir ${SAVE} --device_ids "${device_ids}" \
      --unfreeze_params ef_ --pruner ${pruner} --sparsity ${sparsity} \
      --attn_mode ${attn_mode} --attn_option ${attn_option} --attn_bn ${attn_bn} \
      --ffn_mode ${ffn_mode} --ffn_option ${ffn_option} --ffn_bn ${ffn_bn} \
      --num_train_epochs ${num_train_epochs} --learning_rate ${learning_rate} \
      --weight_decay ${weight_decay} --metric_for_best_model ${metric_for_best_model} \
      --val_max_target_length 60 --max_eval_samples 1600 \
      --num_beams 6 --max_length 60 --min_length 10 --no_repeat_ngram_size 3 \
      > ${SAVE}/log.out & echo $! > ${SAVE}/log.txt &