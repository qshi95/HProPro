

START=0
END=200
echo START:${START}
echo END:${END}

DATA_NAME="hybridqa_dev"
MODEL="gpt-4"
SHOT_NUM=4

# Don't Change these
DATA_PATH="data/${DATA_NAME}.json"

DATETIME_SUFFIX=$(date +'%m-%d_%H-%M')

OUTPUT_PATH_BASE="outputs"
OUTPUT_PATH="${DATETIME_SUFFIX}_${DATA_NAME}_${MODEL}_s${START}_e${END}_mt_${SHOT_NUM}shot"

RUN_DIR=${OUTPUT_PATH_BASE}/${OUTPUT_PATH}

echo ""${RUN_DIR}
mkdir -p ${RUN_DIR}

cp "$0" "${RUN_DIR}/run.sh"

python hybridqa/run_hybridqa_multi_threads.py\
    --task_name ${DATA_NAME}\
    --start ${START}\
    --end ${END}\
    --shot_num ${SHOT_NUM}\
    --data_path ${DATA_PATH}\
    --output_path ${RUN_DIR}\
    --model ${MODEL}\
    --dump_output\
    --simplify\
    --reflection
