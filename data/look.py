import h5py
import numpy as np
# 打开 HDF5 文件
file_path = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/rdt/rdt-ft-data/new_final/episode_301.hdf5"  # 替换为你的文件路径
with h5py.File(file_path, 'r') as f:
    # 打印文件中的所有主键
    def print_attrs(name, obj):
        print(f"Object: {name}")
        for key, value in obj.attrs.items():
            print(f"  {key}: {value}")
    
    # 打印文件中各个组和数据集的结构
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
    
    f.visititems(print_structure)
    f.visititems(print_attrs)

