import os
import h5py

def clean_hdf5(file_path):
    """ 删除 HDF5 文件中不需要的数据 """
    with h5py.File(file_path, 'r+') as f:
        keys_to_remove = [
            'base_action',
            'observations/effort',
            'observations/qvel'
        ]
        
        for key in keys_to_remove:
            if key in f:
                del f[key]
                print(f"Removed: {key} from {file_path}")
        
        print(f"Processed {file_path}\n")

def process_hdf5_in_directory(directory):
    """ 遍历文件夹并处理所有 HDF5 文件 """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                clean_hdf5(file_path)

# 指定 HDF5 文件夹路径
hdf5_directory = "/home/agilex/RoboticsDiffusionTransformer-main/data/datasets/my_cool_dataset"  # 修改为你的实际文件夹路径
process_hdf5_in_directory(hdf5_directory)
