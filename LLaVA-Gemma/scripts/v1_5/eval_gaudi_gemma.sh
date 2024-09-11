#!/bin/bash

export PYTHONPATH=$PWD

# Limits internal graph size to 1000 Ops and reduces the lazy mode memory overheard.
# This will be improved in future releases. Note: This may affect performance.
#export PT_HPU_MAX_COMPOUND_OP_SIZE=1
#export PT_HPU_MAX_COMPOUND_OP_SIZE=1000
#export PT_HPU_MAX_COMPOUND_OP_SIZE=2000
# Sets memory pool to consume the entire HBM memory.
#export PT_HPU_POOL_MEM_ACQUIRE_PERC=100

#export PT_HPU_RECIPE_CACHE_CONFIG=$PWD/pt_cache,True,20000
#export PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1

export ENABLE_CONSOLE=true
export LOG_LEVEL=3

# Export the LOCAL_RANK_MAP environment variable
export LOCAL_RANK_MAP=$(hl-smi -Q module_id -f csv | tail -n +2 | tr '\n' ',' | sed 's/,$//')

NUM_HPU=$(echo "$LOCAL_RANK_MAP" | awk -v col=1 '{$col=gsub(",", "", $col)+1; print}')
# Supported families are "vqa-v2", "gqa", "pope"
DATASET_FAMILY=${DATASET_FAMILY:-gqa}
# Typical variants include "full", "ocr-full", "subsampled", and "slim"
DATASET_VARIANT=${DATASET_VARIANT:-full}
DATASET_DIR="/data0/visual-llama/datasets/vlm-evaluation"
MODEL_DIR="/data0/visual-llama/checkpoints/llava-gemma-2b-it-finetune/checkpoint-5196"
MODEL_ID="llava-gemma-v1.5"

echo "Number of HPUs:    $NUM_HPU"
echo "Evaluation:        $DATASET_FAMILY-$DATASET_VARIANT"
echo "Dataset Directory: $DATASET_DIR"
echo "Model Directory:   $MODEL_DIR"
echo "Model ID:          $MODEL_ID"

EVAL_COMMANDS="-m llava.eval.evaluate --model_dir $MODEL_DIR --model_id $MODEL_ID --dataset.type $DATASET_FAMILY-$DATASET_VARIANT --dataset.root_dir $DATASET_DIR"
if [[ "$NUM_HPU" != "1" ]]; then
    ACCELERATE_COMMANDS="--num_processes=$NUM_HPU --multi_gpu"
else
    ACCELERATE_COMMANDS="--num_processes=1"
fi

if [ ! -d "$DATASET_DIR/datasets/$DATASET_FAMILY" ]; then
    echo "Prepare Command: python /workspace/vlm-evaluation/scripts/datasets/prepare.py --dataset_family $DATASET_FAMILY --root_dir $DATASET_DIR"
    python /workspace/vlm-evaluation/scripts/datasets/prepare.py --dataset_family $DATASET_FAMILY --root_dir $DATASET_DIR
    [ $? -eq 1 ] && exit 1
else
    echo "Dataset '$DATASET_FAMILY' already exists at '$DATASET_DIR'"
    echo "Skipping download of dataset."
fi
echo "Eval Command: accelerate launch $ACCELERATE_COMMANDS $EVAL_COMMANDS"
accelerate launch $ACCELERATE_COMMANDS $EVAL_COMMANDS
[ $? -eq 1 ] && exit 1
echo "Score Command: python /workspace/vlm-evaluation/scripts/score.py --dataset.type $DATASET_FAMILY-$DATASET_VARIANT --dataset.root_dir $DATASET_DIR" --model_id $MODEL_ID
python /workspace/vlm-evaluation/scripts/score.py --dataset.type $DATASET_FAMILY-$DATASET_VARIANT --dataset.root_dir $DATASET_DIR --model_id $MODEL_ID
[ $? -eq 1 ] && exit 1
