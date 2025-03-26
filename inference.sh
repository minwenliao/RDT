#python -m scripts.agilex_inference \
#    --use_actions_interpolation \

#   --pretrained_model_name_or_path="checkpoints/your_finetuned_ckpt.pt" \  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt,
#    --lang_embeddings_path="outs/lang_embeddings/your_instr.pt" \
#   --ctrl_freq=25    # your control frequency

python -m scripts.agilex_inference \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="checkpoints/rdt-1b" \
    --lang_embeddings_path="outs/handover_pan.pt" \
    --publish_rate=25 \
    --ctrl_freq=25

