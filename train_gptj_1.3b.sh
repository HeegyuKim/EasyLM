
# Train GPT-J 1.3B with jax pjit

# batch size is 512K tokens (4batch * 128acc * 1Kseq)

# logger, dataset, model, training_params, checkpoint


    
python3 -m EasyLM.models.gptj.gptj_train \
    --initialize_jax_distributed=true \
    --mp_mesh_dim="8" \
    --tokenizer.name "heegyu/kogpt-j-1.3b" \
    --load_gptj_config "huggingface::heegyu/kogpt-j-1.3b" \
    --logger.project 'gpt2' \
    --total_steps 1000 \
    --save_model_freq=1000 \
    --optimizer.type=adamw \
    --optimizer.adamw_optimizer.lr=1e-4 \
    --optimizer.accumulate_gradient_steps=128 \
    --checkpointer.float_dtype='fp32' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.type='huggingface' \
    --train_dataset.huggingface_dataset.path='heegyu/kowikitext' \
    --train_dataset.huggingface_dataset.streaming=true \
    --train_dataset.huggingface_dataset.batch_size=4  \
    --train_dataset.huggingface_dataset.name='20221001' \
    --train_dataset.huggingface_dataset.split='train' 
    
    # --output_dir=/data/checkpoint/gptj-large/ 
