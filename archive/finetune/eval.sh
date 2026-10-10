export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RootDir=$RootDir/..

Dataset=${2:-amrbart-new}
DataPath=$RootDir/ds/$Dataset

Model=$1
ModelCache=$RootDir/.cache
DataCache=$DataPath/.cache/dump-amrparsing

OutputDir=${RootDir}/outputs/eval-$Model

if [ ! -d ${OutputDir} ];then
  mkdir -p ${OutputDir}
fi

export HF_DATASETS_CACHE=$DataCache

if [ ! -d ${DataCache} ];then
  mkdir -p ${DataCache}
fi

lr=1e-6
batch_size=5

python -u main.py \
    --data_dir $DataPath \
    --task "text2amr" \
    --test_file $DataPath/test.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --overwrite_cache True \
    --model_name_or_path $RootDir/models/$Model \
    --overwrite_output_dir \
    --unified_input True \
    --per_device_eval_batch_size $batch_size \
    --max_source_length 400 \
    --max_target_length 1024 \
    --val_max_target_length 1024 \
    --generation_max_length 1024 \
    --generation_num_beams 5 \
    --predict_with_generate \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --seed 42 \
    --dataloader_num_workers 1 \
    --eval_dataloader_num_workers 1 \
    --include_inputs_for_metrics \
    --metric_for_best_model "eval_smatch" \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run.log