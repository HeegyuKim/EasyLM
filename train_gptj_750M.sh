
# Train GPT-J 750M with jax pjit
# batch size is 512K tokens (4batch * 128acc * 1Kseq)
# logger, dataset, model, training_params, checkpoint

TRAIN_TOKENS=20000000000 # 20B
DEVICES=2 # DP=2, MP=4
SEQ_LEN=1024
BATCH_SIZE=512
MICROBATCH_SIZE=4
TOTAL_STEPS_PER_DEVICE=$((TRAIN_TOKENS / SEQ_LEN / DEVICES / MICROBATCH_SIZE)) # 20B / 1024 tokens / 1 device / 4 mb = 5M
ACCUM_STEPS=$((BATCH_SIZE / MICROBATCH_SIZE / DEVICES)) # 512 / 4 / 1 = 128

save_model_freq=$((TOTAL_STEPS_PER_DEVICE / 5)) # 4B마다 저장
warmup_steps=50000

echo "total step", $TOTAL_STEPS_PER_DEVICE
python3 -m EasyLM.models.gptj.gptj_train \
    --initialize_jax_distributed=true \
    --mp_mesh_dim="4" \
    --tokenizer.name "heegyu/kogpt-j-large" \
    --load_gptj_config "huggingface::heegyu/kogpt-j-large" \
    --logger.online true \
    --logger.project 'gpt2' \
    --logger.experiment_id 'kogpt-j-large' \
    --logger.output_dir "./wandb/" \
    --total_steps $TOTAL_STEPS_PER_DEVICE \
    --save_model_freq=$save_model_freq \
    --optimizer.type=adamw \
    --optimizer.adamw_optimizer.lr=4e-4 \
    --optimizer.adamw_optimizer.end_lr=4e-5 \
    --optimizer.adamw_optimizer.lr_decay_steps=$((TOTAL_STEPS_PER_DEVICE-warmup_steps)) \
    --optimizer.adamw_optimizer.lr_warmup_steps=$warmup_steps \
    --optimizer.accumulate_gradient_steps=$ACCUM_STEPS \
    --checkpointer.float_dtype='fp32' \
    --train_dataset.type='encoded-json' \
    --train_dataset.encoded_json_dataset.path='/data/v1-vocab51k-block1024/*' \
    --train_dataset.encoded_json_dataset.cache_dir='/data/.cache/' \
    --train_dataset.encoded_json_dataset.batch_size=$MICROBATCH_SIZE
    