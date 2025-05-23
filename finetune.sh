# 禁用 NCCL 的 InfiniBand 支持（NCCL 对单卡训练不需要分布式配置）
export NCCL_IB_DISABLE=1  # 禁用 InfiniBand（对于单卡训练，不需要）
export NCCL_SOCKET_IFNAME=lo  # 设置网络接口为 "lo"（回环接口）
export NCCL_DEBUG=INFO  # 设置 NCCL 调试级别为 INFO，以查看调试信息

# 不需要设置 NCCL_NVLS_ENABLE 相关的值
# export NCCL_NVLS_ENABLE=1  # 如果使用分布式时可以启用，但对于单卡不需要
#export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
#export NCCL_IB_DISABLE=0
#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFO
#export NCCL_NVLS_ENABLE=0
# 禁用分布式训练，只使用单卡 GPU
#export CUDA_VISIBLE_DEVICES=6 # 确保只使用 GPU 0（如果你有多个 GPU，指定你要使用的卡）

# 其他环境变量设置
export TEXT_ENCODER_NAME="google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/3_14"
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
    --checkpointing_period=5000 \
    --sample_period=500 \
    --checkpoints_total_limit=40 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=4 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_hdf5 \
    --report_to=tensorboard \
    #--precomp_lang_embed


    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    
