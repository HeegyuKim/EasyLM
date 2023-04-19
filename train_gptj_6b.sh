# Train GPT-J 1.3B with jax pjit

# batch size is 512K tokens (4batch * 128acc * 1Kseq)

# logger, dataset, model, training_params, checkpoint


MODEL="EleutherAI/gpt-j-6b"
    
python3 -m EasyLM.models.gptj.gptj_train \
    --initialize_jax_distributed=true \
    --mp_mesh_dim="8" \
    --tokenizer.name $MODEL \
    --load_gptj_config "huggingface::$MODEL" \
    --logger.project 'gpt2' \
    --logger.output_dir '' \
    --total_steps 1000 \
    --save_model_freq=1000 \
    --optimizer.type=adamw \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --optimizer.accumulate_gradient_steps=512 \
    --checkpointer.float_dtype='fp32' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.type='huggingface' \
    --train_dataset.huggingface_dataset.path='heegyu/kowikitext' \
    --train_dataset.huggingface_dataset.streaming=true \
    --train_dataset.huggingface_dataset.batch_size=1  \
    --train_dataset.huggingface_dataset.name='20221001' \
    --train_dataset.huggingface_dataset.split='train' 
    
    # --output_dir=/data/checkpoint/gptj-large/ 
