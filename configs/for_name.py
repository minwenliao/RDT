import os
import json

# 指定父文件夹路径
parent_folder =  "/baai-cwm-1/baai_cwm_ml/public_data/scenes/rdt/rdt-ft-data/rdt_data"

# 获取所有子文件夹名称
subfolders = [folder for folder in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder))]

# 在每个子文件夹名称后加上 ': 50'
subfolders_with_50 = {folder: 50 for folder in subfolders}

# 将子文件夹名称和对应的值存入 JSON 文件
output_file = "/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/configs/fr.json"
with open(output_file, 'w') as f:
    json.dump(subfolders_with_50, f, indent=4)

print(f"子文件夹名称和对应的值已经保存到 {output_file}")
