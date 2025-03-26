export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_SOCKET_IFNAME=lo,eth0,eth1,eth2,eth3,eth4,eth5,eth6,ciliumhost@eth0,ipvl14@eth6,eth7
export NCCL_IB_DISABLE=0  # 启用 InfiniBand
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=1

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

deepspeed --include localhost:0,1,2,3,4,5,6,7 main.py \
    --deepspeed="./configs/zero2.json" \
    --pretrained_model_name_or_path="./google/rdt-1b" \
    --pretrained_text_encoder_name_or_path=google/t5-v1_1-xxl \
    --pretrained_vision_encoder_name_or_path=google/siglip-so400m-patch14-384 \
    --output_dir=./checkpoints/rdt_2_22 \
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