#!/bin/bash

values=("50" "100" "150" "200" "250" "300" "350" "400" "450" "500")
for i in "${values[@]}"
do
model_name="/home/wexiong_google_com/wx/dpo_train/Online-RLHF/dpo_iter1_sft1/checkpoint-${i}"
#/home/wexiong_google_com/wx/dpo_train/Online-RLHF/dpo_iter1_sft1
#model_name="1231czx/2b_sft1"
bash run_8gpu.sh $model_name
sleep 30
#/home/wexiong_google_com/wx/dpo_train/Online-RLHF/dpo_plus_1nll_iter1
bash /home/wexiong_google_com/wx/ToRA/src/scripts/infer.sh $model_name

pkill -f "python -m vllm.entrypoints.api_server"


done
