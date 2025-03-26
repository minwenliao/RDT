import h5py
import os

# Define the path to your HDF5 file
hdf5_file_path = "/home/agilex/RoboticsDiffusionTransformer-main/data/datasets/0124_1/aloha_mobile_dummy/episode_3.hdf5"

# Required datasets
required_datasets = {
    "action": None,
    #"instruction": None,
    "observations/qpos": None,
    "observations/images/cam_high": None,
    "observations/images/cam_left_wrist": None,
    "observations/images/cam_right_wrist": None
}

# Datasets to be removed
unnecessary_datasets = ["base_action", "observations/effort", "observations/qvel"]

def explore_hdf5(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    with h5py.File(file_path, 'r') as f:
        print("\n--- Dataset Details ---\n")

        # Traverse the HDF5 file structure
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape, dtype = obj.shape, obj.dtype
                required_datasets[name] = (shape, dtype) if name in required_datasets else None
                print(f"Dataset: {name}, shape: {shape}, dtype: {dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")

        f.visititems(visit_func)

        print("\n--- Objects ---\n")
        for key in required_datasets.keys():
            print(f"Object: {key}")

        # Unnecessary datasets
        print("\n--- Unnecessary Datasets (Can be Removed) ---")
        for ds in unnecessary_datasets:
            if ds in f:
                print(f"Unnecessary: {ds} (Exists, should be removed)")

if __name__ == "__main__":
    explore_hdf5(hdf5_file_path)
#用这个代码处理hdf5中不必要的格式最后变为
'''Dataset: action, shape: (300, 14), dtype: float32
Dataset: instruction, shape: (), dtype: object
Group: observations
Group: observations/images
Dataset: observations/images/cam_high, shape: (300, 480, 640, 3), dtype: uint8
Dataset: observations/images/cam_left_wrist, shape: (300, 480, 640, 3), dtype: uint8
Dataset: observations/images/cam_right_wrist, shape: (300, 480, 640, 3), dtype: uint8
Dataset: observations/qpos, shape: (300, 14), dtype: float32
Object: action
Object: instruction
Object: observations
Object: observations/images
Object: observations/images/cam_high
Object: observations/images/cam_left_wrist
Object: observations/images/cam_right_wrist
Object: observations/qposh'''