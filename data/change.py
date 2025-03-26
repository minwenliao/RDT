import h5py
import os

# 定义根目录
root_dir = '/home/agilex/RoboticsDiffusionTransformer-main/data/datasets/0124_1/aloha_mobile_dummy/'

# 定义目标指令
target_instruction = b"Pick up the red bag on the right and put it into the yellow packaging box on the left."

# 遍历目录中的所有 HDF5 文件
for root, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith('.hdf5'):
            filepath = os.path.join(root, filename)
            try:
                # 以读写模式打开 HDF5 文件
                with h5py.File(filepath, 'a') as f:
                    # 检查是否已经存在 'instruction' 数据集
                    if 'instruction' not in f:
                        # 添加 'instruction' 数据集
                        f.create_dataset('instruction', data=target_instruction)
                        print(f"'instruction' dataset added to {filepath}.")
                    else:
                        # 如果 'instruction' 数据集已存在，更新其值
                        f['instruction'][()] = target_instruction
                        print(f"'instruction' dataset updated in {filepath}.")
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")