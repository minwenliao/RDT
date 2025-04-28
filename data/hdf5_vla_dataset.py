import os
import fnmatch
import h5py
import yaml
import cv2
import numpy as np
from configs.state_vec import STATE_VEC_IDX_MAPPING
import pprint  # 用于格式化输出字典
import json
import random

class HDF5VLADataset:
    """
    该类用于从存储在 HDF5 中的体现数据集（embodiment dataset）中采样集。
    """
    def __init__(self, config) -> None:
        # [修改] HDF5 数据集目录的路径
        # 每个 HDF5 文件包含一个 episode
        # HDF5_DIR = "/baai-cwm-nas/public_data/scenes/ych_newpick/push_button/"
        # self.DATASET_NAME = "push_button"
        self.DATASET_NAME = config['dataset']['name']
        HDF5_DIR = f"/baai-cwm-nas/public_data/scenes/ych_newpick/{self.DATASET_NAME}/"
        
        self.file_paths = []
        for root, _, files in os.walk(HDF5_DIR):
            for filename in fnmatch.filter(files, '*.hdf5'):
                file_path = os.path.join(root, filename)
                try:
                    h5py.File(file_path,'r')
                except OSError:
                    print("ignore", file_path)
                    continue
                self.file_paths.append(file_path)
                
        # Load the config
        # with open('configs/base.yaml', 'r') as file:
        #     config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        self.effort_type = config['model']['effort_type']
        self.effort_dim = config['model']['effort_dim']
        if self.effort_type in ["his_c", "his_c_fut"]:
            self.effort_history_steps = tuple(4*i-36 for i in range(10))
        # if self.effort_type == "state":
        #     pass
        #     # self.DATASET_NAME += "_effort_in_state"
        # elif self.effort_type == "his_c":
        #     # self.DATASET_NAME += "_effort_his_c"
        #     # Define relative timesteps for historical effort collection
        #     self.effort_history_steps = tuple(4*i-36 for i in range(10))
        # elif self.effort_type == "fut":
        #     pass
        #     # self.DATASET_NAME += "_effort_fut"
        # elif self.effort_type == "his_c_fut":
        #     pass
        # Get each episode's len
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            _len = res['state'].shape[0] if valid else 0
            episode_lens.append(_len)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.file_paths)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False):
        """Get a training sample at a random timestep.

        Args:
            index (int, optional): the index of the episode.
                If not provided, a random episode will be selected.
            state_only (bool, optional): Whether to return only the state.
                In this way, the sample will contain a complete trajectory rather
                than a single timestep. Defaults to False.

        Returns:
           sample (dict): a dictionary containing the training sample.
        """
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = self.parse_hdf5_file(file_path) \
                if not state_only else self.parse_hdf5_file_state_only(file_path)
            if valid:
                return sample
            else:
                index = np.random.randint(0, len(self.file_paths))
    
    def parse_hdf5_file(self, file_path):
        """[Modify] Parse a hdf5 file to generate a training sample at
            a random timestep.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "meta": {
                        "dataset_name": str,    # the name of your dataset.
                        "#steps": int,          # the number of steps in the episode,
                                                # also the total timesteps.
                        "instruction": str      # the language instruction for this episode.
                    },                           
                    "step_id": int,             # the index of the sampled step,
                                                # also the timestep t.
                    "state": ndarray,           # state[t], (1, STATE_DIM).
                    "state_std": ndarray,       # std(state[:]), (STATE_DIM,).
                    "state_mean": ndarray,      # mean(state[:]), (STATE_DIM,).
                    "state_norm": ndarray,      # norm(state[:]), (STATE_DIM,).
                    "actions": ndarray,         # action[t:t+CHUNK_SIZE], (CHUNK_SIZE, STATE_DIM).
                    "state_indicator", ndarray, # indicates the validness of each dim, (STATE_DIM,).
                    "cam_high": ndarray,        # external camera image, (IMG_HISORY_SIZE, H, W, 3)
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_high_mask": ndarray,   # indicates the validness of each timestep, (IMG_HISORY_SIZE,) boolean array.
                                                # For the first IMAGE_HISTORY_SIZE-1 timesteps, the mask should be False.
                    "cam_left_wrist": ndarray,  # left wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                    "cam_left_wrist_mask": ndarray,
                    "cam_right_wrist": ndarray, # right wrist camera image, (IMG_HISORY_SIZE, H, W, 3).
                                                # or (IMG_HISORY_SIZE, 0, 0, 0) if unavailable.
                                                # If only one wrist, make it right wrist, plz.
                    "cam_right_wrist_mask": ndarray
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # We randomly sample a timestep
            step_id = np.random.randint(first_idx-1, num_steps)
            
            # Load the instruction
            dir_path = os.path.dirname(file_path)
            #with open(os.path.join(dir_path, 'expanded_instruction_gpt-4-turbo.json'), 'r') as f_instr:
             #   instruction_dict = json.load(f_instr)
            # We have 1/3 prob to use original instruction,
            # 1/3 to use simplified instruction,
            # and 1/3 to use expanded instruction.
            #instruction_type = np.random.choice([
             #  'instruction', 'simplified_instruction', 'expanded_instruction'])
            #instruction = instruction_dict[instruction_type]
            #if isinstance(instruction, list):
             #   instruction = np.random.choice(instruction)
            # You can also use precomputed language embeddings (recommended)
            #instruction =  "Grasp the mineral water bottle that is located on the table by securely wrapping your fingers around its body, and then tilt it gently to pour water into the mug, stopping once the water level reaches approximately halfway up the sides of the mug.", "Carefully grasp the mineral water bottle located on the table and gently tilt it to pour the water into the mug, stopping when the water level reaches halfway up the sides of the mug."
            # instruction ="Use the robotic arm to perform a precise and controlled motion, utilizing the gripper to gently push the blocks over. Ensure that all movements are smooth and stable, avoiding sudden acceleration or abrupt contact. The arm should approach the block slowly, aligning the gripper at a slight angle to the block’s surface. Then, apply a gradual horizontal force through the gripper to push the block until it tips over in a controlled manner. Throughout the operation, continuously monitor the arm's position, velocity, and applied force to maintain safety, stability, and consistency of the motion."
            # instruction = f"output/{self.DATASET_NAME}.pt"
            instructions = {"pick_up_glass_rod3":[
                "With the right arm of the robotic arm, gently and steadily grasp the glass rod by its midsection, ensuring a secure grip, and then lift it slowly to prevent any instability.",
                "Carefully use the right arm of the robotic arm to grasp the glass rod, making sure to secure it without applying too much force. Once the rod is firmly held, lift it gently and steadily to avoid any risk of breakage.",
                "Using the robotic arm’s right hand, softly grasp the glass rod by its lower end, applying just enough pressure to hold it securely. Slowly and steadily, lift the rod, maintaining a smooth and controlled motion to prevent any abrupt movements.",
                "With great precision, use the right arm of the robotic arm to gently grasp the glass rod. Ensure the grip is both secure and light, then begin lifting it carefully, ensuring minimal force is applied to avoid any unintended jerking or slipping."
            ],
            "push_button":[
                "With the right arm of the robotic arm, gently and steadily grasp the glass rod by its midsection, ensuring a secure grip, and then lift it slowly to prevent any instability.",
                "Carefully use the right arm of the robotic arm to grasp the glass rod, making sure to secure it without applying too much force. Once the rod is firmly held, lift it gently and steadily to avoid any risk of breakage.",
                "Using the robotic arm’s right hand, softly grasp the glass rod by its lower end, applying just enough pressure to hold it securely. Slowly and steadily, lift the rod, maintaining a smooth and controlled motion to prevent any abrupt movements.",
                "With great precision, use the right arm of the robotic arm to gently grasp the glass rod. Ensure the grip is both secure and light, then begin lifting it carefully, ensuring minimal force is applied to avoid any unintended jerking or slipping."
            ], #由于训练疏忽它和pick_up_glass_rod3是相同的
            "plug_charger": [
                "With the dual-prong charger held in the right robotic arm, carefully align it with the power strip on the table and plug it in securely.",
                "Using the right robotic arm, precisely position the dual-prong charger into the power strip on the table, ensuring a firm connection.",
                "With the right robotic arm, carefully guide the dual-prong charger into the power strip on the table and plug it in securely.",
                "Using the right robotic arm, gently but firmly insert the dual-prong charger into the power strip on the table.",
                "With the dual-prong charger in the right robotic arm, carefully align and plug it into the power strip on the table.",
            ]
            }[self.DATASET_NAME]

            # Randomly select one instruction
            instruction = np.random.choice(instructions)

            # Assemble the meta
            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": instruction
            }
            
            # Rescale gripper to [0, 1]
            qpos = qpos * np.array(
               [[1, 1, 1, 1, 1, 1, 10.6157, 1, 1, 1, 1, 1, 1, 10.6157]] 
            )
            target_qpos = f['action'][step_id:step_id+self.CHUNK_SIZE] * np.array(
               [[1, 1, 1, 1, 1, 1, 10.6157, 1, 1, 1, 1, 1, 1, 10.6157]] 
            )
            
            # Parse the state and action
            state = qpos[step_id:step_id+1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos**2, axis=0))

            effort = None
            effort_std = None
            effort_mean = None
            effort_norm = None
            if self.effort_type == "state":
                effort = f['observations']['effort'][step_id:step_id+1]
                effort_std = np.std(effort, axis=0)
                effort_mean = np.mean(effort, axis=0)
                effort_norm = np.sqrt(np.mean(effort**2, axis=0))
            elif self.effort_type == "his_c":
                effort = self.get_effort_history(f, step_id)
            elif self.effort_type == "fut":
                effort = f['observations']['effort'][step_id:step_id+self.CHUNK_SIZE]
                if len(effort) < self.CHUNK_SIZE:
                    # Pad with the last effort
                    effort = np.concatenate([
                        effort,
                        np.tile(effort[-1:], (self.CHUNK_SIZE - len(effort), 1))
                    ], axis=0)
            elif self.effort_type == "his_c_fut":
                # Combine both historical and future effort
                history_effort = self.get_effort_history(f, step_id)
                future_effort = f['observations']['effort'][step_id:step_id+self.CHUNK_SIZE]
                if len(future_effort) < self.CHUNK_SIZE:
                    # Pad with the last effort
                    future_effort = np.concatenate([
                        future_effort,
                        np.tile(future_effort[-1:], (self.CHUNK_SIZE - len(future_effort), 1))
                    ], axis=0)
                effort = (history_effort, future_effort)
            
            actions = target_qpos
            if actions.shape[0] < self.CHUNK_SIZE:
                # Pad the actions using the last action
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE-actions.shape[0], 1))
                ], axis=0)
            
            # Fill the state/action into the unified vector
            def fill_in_state(values, effort=None):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
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
                if self.effort_type == "state":
                    EFFORT_INDICES = list(range(103, 103 + self.effort_dim)) # reserved
                    uni_vec[..., EFFORT_INDICES] = effort
                return uni_vec
            state = fill_in_state(state, effort)
            state_indicator = fill_in_state(np.ones_like(state_std), np.ones_like(effort_std))
            state_std = fill_in_state(state_std, effort_std)
            state_mean = fill_in_state(state_mean, effort_mean)
            state_norm = fill_in_state(state_norm, effort_norm)
            # If action's format is different from state's,
            # you may implement fill_in_action()
            actions = fill_in_state(actions)
            
            if self.effort_type == "fut":
                # Concatenate effort after the unified 128-dim action space
                actions = np.concatenate([actions, effort], axis=1)
                # state = np.concatenate([state, np.zeros((1, self.effort_dim), dtype=state.dtype)], axis=1)
            elif self.effort_type == "his_c_fut":
                history_effort, future_effort = effort
                # state = np.concatenate([state, np.zeros((1, self.effort_dim), dtype=state.dtype)], axis=1)
                # state_indicator = np.concatenate([state_indicator, np.zeros((1, self.effort_dim), dtype=state_indicator.dtype)], axis=1)
                actions = np.concatenate([actions, future_effort], axis=1)
                effort = history_effort
            
            # Parse the images
            def parse_img(key):
                imgs = []
                for i in range(max(step_id-self.IMG_HISORY_SIZE+1, 0), step_id+1):
                    img = f['observations']['images'][key][i]
                    #imgs.append(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR))
                    imgs.append(img)
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    # Pad the images using the first image
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE-imgs.shape[0], 1, 1, 1)),
                        imgs
                    ], axis=0)
                return imgs
            # `cam_high` is the external camera image
            cam_high = parse_img('cam_high')
            # For step_id = first_idx - 1, the valid_len should be one
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len
            )
            cam_left_wrist = parse_img('cam_left_wrist')
            cam_left_wrist_mask = cam_high_mask.copy()
            cam_right_wrist = parse_img('cam_right_wrist')
            cam_right_wrist_mask = cam_high_mask.copy()
            
            # Return the resulting sample
            # For unavailable images, return zero-shape arrays, i.e., (IMG_HISORY_SIZE, 0, 0, 0)
            # E.g., return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)) for the key "cam_left_wrist",
            # if the left-wrist camera is unavailable on your robot
            sample = {
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
            if self.effort_type not in ["state", "no"]:
                sample["effort"] = effort
            return True, sample

    def get_effort_history(self, f, step_id):
        """Get effort values from specific historical timesteps.
        
        Args:
            f: HDF5 file object
            step_id: Current step ID
            
        Returns:
            numpy array: Concatenated effort values from historical timesteps
        """
        efforts = []
        for rel_step in self.effort_history_steps:
            abs_step = step_id + rel_step
            # Pad with edge values if step is out of bounds
            if abs_step < 0:
                effort = f['observations']['effort'][0:1]
            elif abs_step >= len(f['observations']['effort']):
                effort = f['observations']['effort'][-1:]
            else:
                effort = f['observations']['effort'][abs_step:abs_step+1]
            efforts.append(effort)
        return np.concatenate(efforts, axis=1)  # Concatenate along feature dimension

    def parse_hdf5_file_state_only(self, file_path):
        """[Modify] Parse a hdf5 file to generate a state trajectory.

        Args:
            file_path (str): the path to the hdf5 file
        
        Returns:
            valid (bool): whether the episode is valid, which is useful for filtering.
                If False, this episode will be dropped.
            dict: a dictionary containing the training sample,
                {
                    "state": ndarray,           # state[:], (T, STATE_DIM).
                    "action": ndarray,          # action[:], (T, STATE_DIM).
                } or None if the episode is invalid.
        """
        with h5py.File(file_path, 'r') as f:
            qpos = f['observations']['qpos'][:]
            num_steps = qpos.shape[0]
            # [Optional] We drop too-short episode
            if num_steps < 128:
                return False, None
            
            # [Optional] We skip the first few still steps
            EPS = 1e-2
            # Get the idx of the first qpos whose delta exceeds the threshold
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            if len(indices) > 0:
                first_idx = indices[0]
            else:
                raise ValueError("Found no qpos that exceeds the threshold.")
            
            # Rescale gripper to [0, 1]
            qpos = qpos / np.array(
               [[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]] 
            )
            target_qpos = f['action'][:] / np.array(
               [[1, 1, 1, 1, 1, 1, 11.8997, 1, 1, 1, 1, 1, 1, 13.9231]] 
            )
            
            # Parse the state and action
            state = qpos[first_idx-1:]
            action = target_qpos[first_idx-1:]

            if self.effort_type == "state":
                effort = f['observations']['effort'][first_idx-1:]
            else:
                effort = None
            
            # Fill the state/action into the unified vector
            def fill_in_state(values, effort=None):
                # Target indices corresponding to your state space
                # In this example: 6 joints + 1 gripper for each arm
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
                if self.effort_type == "state":
                    EFFORT_INDICES = list(range(103, 103 + self.effort_dim)) # reserved
                    uni_vec[..., EFFORT_INDICES] = effort
                return uni_vec
            state = fill_in_state(state, effort)
            action = fill_in_state(action)
            
            # Return the resulting sample
            return True, {
                "state": state,
                "action": action
            }

if __name__ == "__main__":
    ds = HDF5VLADataset()
    for i in range(len(ds)):
        print(f"Processing episode {i}/{len(ds)}...")
        ds.get_item(i)        