import os
import json
import sys

# 将项目根目录添加到 Python 的搜索路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import yaml
from tqdm import tqdm



from models.multimodal_encoder.t5_encoder import T5Embedder


GPU = 0
MODEL_PATH = "/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/google/t5-v1_1-xxl"
CONFIG_PATH = "configs/base.yaml"
# Modify the TARGET_DIR to your dataset path
TARGET_DIR = "output"


# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
# OFFLOAD_DIR = "/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/datasets/for_water/"  # Specify your offload directory here, ensuring the directory exists.

def main():
    with open(CONFIG_PATH, "r") as fp:
        config = yaml.safe_load(fp)
    
    device = torch.device(f"cuda:{GPU}")
    text_embedder = T5Embedder(
        from_pretrained=MODEL_PATH, 
        model_max_length=config["dataset"]["tokenizer_max_length"], 
        device=device,
        # use_offload_folder=OFFLOAD_DIR
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
    
    # Assuming you only have one task, specify the task path
    task_path = TARGET_DIR  # Or you can define a specific task directory if needed

    # Load the instructions corresponding to the task
    # with open("/baai-cwm-1/baai_cwm_ml/public_data/scenes/rdt/rdt-ft-data/rdt_data/pour_water_4/expanded_instruction_gpt-4-turbo.json", 'r') as f_instr:
    #     instruction_dict = json.load(f_instr)
    #     instructions = [instruction_dict['instruction']] + instruction_dict['simplified_instruction'] + \
    #     instruction_dict['expanded_instruction']
    instructions=["Push the button."]
    
    # Encode the instructions
    tokenized_res = tokenizer(
        instructions, return_tensors="pt",
        padding="longest",
        truncation=True
    )
    tokens = tokenized_res["input_ids"].to(device)
    attn_mask = tokenized_res["attention_mask"].to(device)
    
    with torch.no_grad():
        text_embeds = text_encoder(
            input_ids=tokens,
            attention_mask=attn_mask
        )["last_hidden_state"].detach().cpu()
    
    attn_mask = attn_mask.cpu().bool()

    # Save the embeddings for training use
    for i in range(len(instructions)):
        text_embed = text_embeds[i][attn_mask[i]]
        save_path = os.path.join(task_path, f"lang_embed_{i}.pt")
        torch.save(text_embed, save_path)

if __name__ == "__main__":
    main()
