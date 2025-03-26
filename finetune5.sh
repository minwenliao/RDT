# 禁用 NCCL 的 InfiniBand 支持（NCCL 对单卡训练不需要分布式配置）
#export NCCL_IB_DISABLE=1  # 禁用 InfiniBand（对于单卡训练，不需要）
#export NCCL_SOCKET_IFNAME=lo  # 设置网络接口为 "lo"（回环接口）
export NCCL_DEBUG=INFO  # 设置 NCCL 调试级别为 INFO，以查看调试信息
#export NVIDIA_DRIVER_CAPABILITIES=all
export CUDA_VISIBLE_DEVICES=0
# 其他环境变量设置
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/rdt_pick"
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

# For run in a single node/machine
# accelerate launch main.py \
#     --deepspeed="./configs/zero2.json" \
#     ...

accelerate launch main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./google/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=16 \
    --sample_batch_size=32 \
    --max_train_steps=200000 \
    --checkpointing_period=1000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard \



    # Use this to resume training from some previous checkpoint
    
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
