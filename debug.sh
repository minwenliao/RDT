# export PATH="/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/rdt/bin:$PATH"
export TEXT_ENCODER_NAME="/baai-cwm-1/baai_cwm_ml/cwm/ziaur.rehman/lmw/code/code/2_3/2/RoboticsDiffusionTransformer-main/t5-v1_1-xxl"
export VISION_ENCODER_NAME="/baai-cwm-1/baai_cwm_ml/cwm/ziaur.rehman/lmw/code/code/rdt/siglip-so400m-patch14-384"
export OUTPUT_DIR="/baai-cwm-nas/algorithm/ziaur.rehman/checkpoints/$3"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="../cutlass"


export WANDB_PROJECT="roboticDiffusionTransformer"


export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
# export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
# export NCCL_TOPO_DUMP_FILE=topo.xml
# export NCCL_IB_DISABLE=0
# #export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO
# export NCCL_NVLS_ENABLE=0

# export NCCL_SOCKET_IFNAME=eth1
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_MIN_NCHANNELS=4
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_TIMEOUT=14


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
    --pretrained_model_name_or_path="/baai-cwm-1/baai_cwm_ml/cwm/ziaur.rehman/lmw/code/code/rdt/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=4 \
    --sample_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --max_train_steps=80000 \
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
    --effort_type="$1" \
    --report_to=tensorboard \
    --dataset="$2"
    

    # --precomp_lang_embed \

    # Use this to resume training from some previous checkpoint
    # --resume_from_checkpoint="checkpoint-36000" \
    # Use this to load from saved lanuage instruction embeddings,
    # instead of calculating it during training
    
