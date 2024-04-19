START=0
END=0
SAMPLE_NUM=0
SEED=-1
MODEL="gpt-4"
SHOT_NUM=4
OUTPUT_PATH="./mmqa_oracle.json"
PEOCESS_NUM=10

python mmqa/run_mmqa_multi_process.py\
    --start=${START}\
    --end=${END}\
    --sample_num=${SAMPLE_NUM}\
    --seed=${SEED}\
    --model=${MODEL}\
    --shot_num=${SHOT_NUM}\
    --reflection\
    --use_caption\
    -o ${OUTPUT_PATH}\
    --process_num=${PEOCESS_NUM}\
    --use_oracle

python mmqa/eval_mmqa.py\
    -o ${OUTPUT_PATH}
