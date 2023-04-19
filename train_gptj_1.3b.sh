
# Train GPT-J 1.3B with jax pjit

# batch size is 512K tokens (4batch * 128acc * 1Kseq)

# logger, dataset, model, training_params, checkpoint



TRAIN_TOKENS=20000000000 # 20B
DEVICES=1 # DP=1, MP=8
SEQ_LEN=1024
BATCH_SIZE=512
MICROBATCH_SIZE=4
TOTAL_STEPS=$((TRAIN_TOKENS / SEQ_LEN / BATCH_SIZE)) # 20M / 512 = 40K
TOTAL_STEPS_PER_DEVICE=$((TOTAL_STEPS / DEVICES)) # 40K / 1 device = 40K
ACCUM_STEPS=$((BATCH_SIZE / MICROBATCH_SIZE / DEVICES)) # 512 / 4 / 1 = 128

save_model_freq=$((TOTAL_STEPS_PER_DEVICE / 5)) # 4B마다 저장

python3 -m EasyLM.models.gptj.gptj_train \
    --initialize_jax_distributed=true \
    --mp_mesh_dim="8" \
    --tokenizer.name "heegyu/kogpt-j-1.3b" \
    --load_gptj_config "huggingface::heegyu/kogpt-j-1.3b" \
    --logger.project 'gpt2' \
    --logger.prefix 'kogpt-j-1.3b' \
    --logger.experiment_id 'kogpt-j-1.3b' \
    --logger.output_dir "./wandb/" \
    --total_steps $TOTAL_STEPS_PER_DEVICE \
    --save_model_freq=$save_model_freq \
    --optimizer.type=adamw \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --optimizer.accumulate_gradient_steps=$ACCUM_STEPS \
    --checkpointer.float_dtype='fp32' \
    --train_dataset.type='encoded-json' \
    --train_dataset.encoded_json_dataset.path='/data/v1-vocab51k-block1024/*' \
    --train_dataset.encoded_json_dataset.cache_dir='/data/.cache/' \
    --train_dataset.encoded_json_dataset.batch_size=$MICROBATCH_SIZE
    
    # --output_dir=/data/checkpoint/gptj-large/ 
