
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi
MODEL_NAME_OR_PATH=$1
#MODEL_NAME_OR_PATH='/home/wexiong_google_com/wx/sft/RLHF-Reward-Modeling/pair-pm/pm_models/gemma-2b-it_bs128_lr1e-5/checkpoint-3511'
#"/home/wexiong_google_com/wx/sft/RLHF-Reward-Modeling/pair-pm/pm_models/gemma_2b_rs_iter1/checkpoint-940"
#"/home/wexiong_google_com/wx/sft/RLHF-Reward-Modeling/pair-pm/pm_models/gemma-2b-it_bs128_lr1e-5/checkpoint-5265"
#"llm-agents/tora-code-34b-v1.0"
#"llm-agents/tora-code-34b-v1.0"
#"llm-agents/tora-code-34b-v1.0"
# MODEL_NAME_OR_PATH="llm-agents/tora-70b-v1.0"

# DATA_LIST = ['math', 'gsm8k', 'gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']

DATA_NAME="math"
#DATA_NAME="gsm8k"

OUTPUT_DIR="./output1"

SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

#--use_train_prompt_format \

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer.infer_api_gemma \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 1 \
--temperature 1 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \

DATA_NAME="gsm8k"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer.infer_api_gemma \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--output_dir ${OUTPUT_DIR} \
--split ${SPLIT} \
--prompt_type ${PROMPT_TYPE} \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 1 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
                      
