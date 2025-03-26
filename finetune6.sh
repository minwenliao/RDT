
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt_2_8_8gpu"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="../cutlass"

export WANDB_PROJECT="robotics_diffusion_transformer"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

deepspeed --hostfile=$MLP_MPI_HOSTFILE main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./google/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --sample_batch_size=64 \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler=constant \
    --learning_rate=1e-4 \
    --mixed_precision=bf16 \
    --dataloader_num_workers=32 \
    --image_aug \
    --dataset_type=finetune \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard 