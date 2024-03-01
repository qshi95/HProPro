export PYTHONPATH=/Volumes/workspace/HPoT/HybridPoT


START=0
END=200
DATA_NAME="hybridqa_dev_200"
MODEL="gpt-4"
SHOT_NUM=4

NOTES="end to end qa"

# Don't Change these

DATA_PATH="data/${DATA_NAME}.json"

DATETIME_SUFFIX=$(date +'%m-%d_%H-%M')

OUTPUT_PATH_BASE="outputs"
OUTPUT_PATH="${DATETIME_SUFFIX}_${DATA_NAME}_${MODEL}_s${START}_e${END}_mt_${SHOT_NUM}shot"

RUN_DIR=${OUTPUT_PATH_BASE}/${OUTPUT_PATH}

echo ""${RUN_DIR}
mkdir -p ${RUN_DIR}

cp "$0" "${RUN_DIR}/run.sh"

python hybridqa/run_hybridqa_multi_threads_end2end.py\
    --task_name ${DATA_NAME}\
    --start ${START}\
    --end ${END}\
    --shot_num ${SHOT_NUM}\
    --data_path ${DATA_PATH}\
    --output_path ${RUN_DIR}\
    --model ${MODEL}\
    --reflection\
    --simplify
