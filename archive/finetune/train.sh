export CUDA_VISIBLE_DEVICES=0
RootDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
RootDir=$RootDir/..

DataPath=$RootDir/ds/$2
ValPath=$RootDir/ds/$3

Model=$1
ModelCache=$RootDir/.cache
DataCache=$DataPath/.cache/dump-amrparsing

OutputDir=${RootDir}/outputs/$Model-fted

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
    --overwrite_output_dir \
    --data_dir $DataPath \
    --task "text2amr" \
    --train_file $DataPath/train.jsonl \
    --validation_file $ValPath/dev.jsonl \
    --test_file $ValPath/test.jsonl \
    --output_dir $OutputDir \
    --cache_dir $ModelCache \
    --data_cache_dir $DataCache \
    --tokenizer_name $RootDir/models/$Model \
    --model_name_or_path $RootDir/models/$Model \
    --unified_input True \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --gradient_accumulation_steps $batch_size \
    --learning_rate $lr \
    --optim "adamw_hf" \
    --lr_scheduler_type "polynomial" \
    --warmup_steps 200 \
    --num_train_epochs 16 \
    --early_stopping 16 \
    --max_source_length 400 \
    --max_target_length 1024 \
    --val_max_target_length 1024 \
    --generation_max_length 1024 \
    --generation_num_beams 5 \
    --label_smoothing_factor 0.1 \
    --evaluation_strategy "epoch" \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --max_steps -1 \
    --predict_with_generate \
    --smart_init False \
    --use_fast_tokenizer False \
    --logging_dir $OutputDir/logs \
    --logging_first_step True \
    --logging_steps 20 \
    --save_steps 10000 \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --seed 42 \
    --dataloader_num_workers 1 \
    --eval_dataloader_num_workers 1 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_smatch" \
    --include_inputs_for_metrics \
    --greater_is_better True \
    --do_train \
    --do_eval \
    --do_predict \
    --ddp_find_unused_parameters False \
    --report_to "tensorboard" \
    --dataloader_pin_memory True 2>&1 | tee $OutputDir/run-$(date +"%Y-%m-%dT%H:%M:%S%:z").log
