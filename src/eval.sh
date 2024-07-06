# run_and_kill.sh
#!/bin/bash
source ~/.bashrc
conda activate tora

values=("100" "150" "200" "250" "300" "350") #"250" "300" "350" "400" "450" "500")
for i in "${values[@]}"
do
model_name="/home/wexiong_google_com/dpo_train/Online-RLHF/mdpo_iter1_gemma7b_lr4e7_bz32_sft_100steps_nll/checkpoint-${i}"
echo "Usage: $model_name <model_path>"
bash run_8gpu.sh $model_name
sleep 150
bash /home/wexiong_google_com/tora/src/scripts/infer.sh $model_name

pkill -f "python -m vllm.entrypoints.api_server"

done
