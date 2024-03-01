export PYTHONPATH=/Volumes/workspace/HPoT/HybridPoT

ID=$1
shift
RUNNING_ARGS="$*"

DATA_NAME="hybridqa_dev"
MODEL="gpt-4"
SHOT_NUM=4
TEMPERATURE=0.7
SAMPLE_N=10

# Don't Change these

DATA_PATH="data/${DATA_NAME}.json"
DATETIME_SUFFIX=$(date +'%m-%d_%H-%M')

OUTPUT_PATH_BASE="single_output"
OUTPUT_PATH="${ID}_${MODEL}_${SHOT_NUM}shot_${DATETIME_SUFFIX}"

RUN_DIR=${OUTPUT_PATH_BASE}/${OUTPUT_PATH}

echo ""${RUN_DIR}
mkdir -p ${RUN_DIR}

cp "$0" "${RUN_DIR}/run.sh"

python hybridqa/run_hybridqa.py\
    --task_name ${ID}\
    --shot_num ${SHOT_NUM}\
    --data_path ${DATA_PATH}\
    --output_path ${RUN_DIR}\
    --model ${MODEL}\
    --id ${ID}\
    --logging\
    --reflection\
    $RUNNING_ARGS
