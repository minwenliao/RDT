import os

import torch
import yaml
import sys
import os

sys.path.append('/baai-cwm-1/baai_cwm_ml/cwm/ziaur.rehman/lmw/code/code/RDT-effort')


from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 0
MODEL_PATH ="/baai-cwm-1/baai_cwm_ml/cwm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
SAVE_DIR = "output"

# Modify this to your task name and instruction
# TASK_NAME = "push_button"
# INSTRUCTION = "With the right arm of the robotic arm, gently and steadily grasp the glass rod by its midsection, ensuring a secure grip, and then lift it slowly to prevent any instability."
# TASK_NAME = "pick_up_glass_rod3"
# INSTRUCTION = "With the right arm of the robotic arm, gently and steadily grasp the glass rod by its midsection, ensuring a secure grip, and then lift it slowly to prevent any instability."
TASK_NAME = "plug_charger"
INSTRUCTION = "With the dual-prong charger held in the right robotic arm, carefully align it with the power strip on the table and plug it in securely.",


# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    tokens = tokenizer(
        INSTRUCTION, return_tensors="pt",
        padding="longest",
        truncation=True
    )["input_ids"].to(device)

    tokens = tokens.view(1, -1)

    
    with torch.no_grad():
        pred = text_encoder(tokens).last_hidden_state.detach().cpu()
    
    save_path = os.path.join(SAVE_DIR, f"{TASK_NAME}.pt")
    # We save the embeddings in a dictionary format
    torch.save({
            "name": TASK_NAME,
            "instruction": INSTRUCTION,
            "embeddings": pred
        }, save_path
    )
    # torch.save(pred, save_path)
    
    print(f'\"{INSTRUCTION}\" from \"{TASK_NAME}\" is encoded by \"{MODEL_PATH}\" into shape {pred.shape} and saved to \"{save_path}\"')


if __name__ == "__main__":
    main()

'''这段代码的目的是将文本指令编码为一个嵌入表示，通常用于自然语言处理（NLP）任务中，例如将文本转换为固定长度的向量表示，以便进一步处理。代码中使用的 T5 模型是一个基于 Transformer 的预训练模型，通常用于多种 NLP 任务，如文本生成、文本分类、翻译等。'''
'''只对单一的任务指令进行编码，将指令通过 T5 模型转化为嵌入。'''