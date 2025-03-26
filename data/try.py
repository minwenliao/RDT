import os
import json
import numpy as np
import h5py
import random
import sys

from tqdm import tqdm
from configs.state_vec import STATE_VEC_IDX_MAPPING
import pprint
import yaml

class HDF5VLADataset:
    """
    该类用于从存储在 HDF5 中的体现数据集（embodiment dataset）中采样集。
    """
    def __init__(self) -> None:
        # [修改] HDF5 数据集目录的路径
        # 每个子文件夹下包含 HDF5 文件和 expanded_instruction_gpt-4-turbo.json 文件，没用到expanded_instruction_None.json
        HDF5_DIR = "/baai-cwm-1/baai_cwm_ml/public_data/scenes/rdt/rdt-ft-data/rdt_data"
        self.DATASET_NAME = None
        self.file_paths = []  # HDF5 文件路径
        self.instruction_dicts = {}  # 每个任务文件夹的指令信息

        for root, dirs, files in os.walk(HDF5_DIR):
            for dir_name in dirs:
                task_dir = os.path.join(root, dir_name)
                json_path = os.path.join(task_dir, "expanded_instruction_gpt-4-turbo.json")
                hdf5_files = [f for f in os.listdir(task_dir) if f.endswith('.hdf5')]

                if os.path.exists(json_path) and hdf5_files:
                    # 加载 JSON 文件中的指令信息
                    with open(json_path, 'r') as f:
                        self.instruction_dicts[dir_name] = json.load(f)

                    # 解析每一个hdf5
                    for hdf5_file in hdf5_files:
                        self.file_paths.append((dir_name, os.path.join(task_dir, hdf5_file)))

        # 加载配置
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        # 获取每个 episode 的长度
        episode_lens = []
        for _, file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)

    def __len__(self):
        return len(self.file_paths)

    def get_item(self, index: int = None, state_only=False):
        """获取一个训练样本，时间步随机选择。"""
        while True:
            if index is None:
                selected_index = np.random.choice(len(self.file_paths), p=self.episode_sample_weights)
                task_name, file_path = self.file_paths[selected_index]
            else:
                task_name, file_path = self.file_paths[index]

            # 动态设置当前数据集名称
            self.DATASET_NAME = task_name
            print(f"Current dataset: {self.DATASET_NAME}")

            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                # 如果 sample 包含 meta 信息，则添加指令
                if 'meta' in sample:
                    # 从对应任务的 JSON 文件中随机选择一个指令
                    instructions = [
                        self.instruction_dicts[task_name]['simplified_instruction'],
                        *self.instruction_dicts[task_name]['expanded_instruction']
                    ]
                    instruction = random.choice(instructions)
                    sample['meta']['instruction'] = instruction
                    sample['meta']['dataset_name'] = task_name

                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))

    def parse_hdf5_file(self, file_path):
        """解析 HDF5 文件并生成训练样本"""
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]

            # 删除太短的 episode
            if num_steps < 128:
                return False, None

            # 跳过前几个静止的步骤
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("未找到超过阈值的 qpos。")

            step_id = np.random.randint(first_idx - 1, num_steps)

            # 构造 meta 信息
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": None  # 初始化为 None，后续在 get_item 中填充
            }

            # 归一化 qpos 和 target_qpos
            qpos = qpos / np.array([[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]])
            target_qpos = f['action'][step_id:step_id + self.CHUNK_SIZE] / np.array(
                [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]]
            )

            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                actions = np.concatenate([
                    actions, 
                    np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))
                ], axis=0)

            # 填充 state/action 信息到统一向量
            def fill_in_state(values):
                UNI_STATE_INDICES = [
                    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["left_gripper_open"]
                ] + [
                    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
                ] + [
                    STATE_VEC_IDX_MAPPING["right_gripper_open"]
                ]
                uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
                uni_vec[..., UNI_STATE_INDICES] = values
                return uni_vec

            state = fill_in_state(state)
            state_indicator = fill_in_state(np.ones_like(state_std))
            state_std = fill_in_state(state_std)
            state_mean = fill_in_state(state_mean)
            state_norm = fill_in_state(state_norm)
            actions = fill_in_state(actions)

            # 解析图像
            def parse_img(key):
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    raw_img = f['observations']['images'][key][i]
                    imgs.append(raw_img)

                imgs = np.stack(imgs, axis=0)  # (时间步数, 480, 640, 3)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # 如果图像数量不够，进行重复填充
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)), 
                        imgs
                    ], axis=0)
                return imgs

            cam_high = parse_img('cam_high')
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array([False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)

            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()

            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()

            # 返回样本
            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                "cam_left_wrist": cam_left_wrist,
                "cam_left_wrist_mask": cam_left_wrist_mask,
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask
            }

    def parse_hdf5_file_state_only(self, file_path):
        """仅提取状态，不涉及图像等额外信息"""
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            return True, {'state': qpos}


def process_and_write_statistics(dataset, output_json_path):
    """
    计算数据集的统计信息并将其写入 JSON 文件
    """
    statistics = {}
    for task_name in dataset.instruction_dicts:
        # 计算每个任务的统计信息
        task_data = []
        for task_idx, (task_name, file_path) in enumerate(dataset.file_paths):
            valid, sample = dataset.get_item(task_idx, state_only=True)
            if valid:
                state = sample['state']
                state_min = np.min(state, axis=0).tolist()
                state_max = np.max(state, axis=0).tolist()
                state_mean = np.mean(state, axis=0).tolist()
                state_std = np.std(state, axis=0).tolist()

                task_data.append({
                    'min': state_min,
                    'max': state_max,
                    'mean': state_mean,
                    'std': state_std,
                })
        
        # 将任务的统计信息加入到整体统计字典中
        statistics[task_name] = task_data

    # 保存统计信息到 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(statistics, f, indent=4)

    print(f"统计信息已保存到 {output_json_path}")


if __name__ == '__main__':
    dataset = HDF5VLADataset()

    # 统计信息保存路径
    output_json_path = "/baai-cwm-1/baai_cwm_ml/algorithm/ziaur.rehman/code/2_3/2/RoboticsDiffusionTransformer-main/data/te.json"
    process_and_write_statistics(dataset, output_json_path)
