from typing import Callable, List, Type
import gymnasium as gym
import numpy as np
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
import argparse
import yaml
import torch
from collections import deque
from PIL import Image
import cv2
import imageio
from functools import partial

from diffusion_policy.workspace.robotworkspace import RobotWorkspace

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1", help=f"Environment to run motion planning solver on. ")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgb", help="Observation mode to use. Usually this is kept as 'none' as observations are not necesary to be stored, they can be replayed later via the mani_skill.trajectory.replay_trajectory script.")
    parser.add_argument("-n", "--num-traj", type=int, default=25, help="Number of trajectories to generate.")
    parser.add_argument("--only-count-success", action="store_true", help="If true, generates trajectories until num_traj of them are successful and only saves the successful trajectories/videos")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be 'sensors' or 'rgb_array' which only affect what is saved to videos")
    parser.add_argument("--vis", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--save-video", action="store_true", help="whether or not to save videos locally")
    parser.add_argument("--traj-name", type=str, help="The name of the trajectory .h5 file that will be created.")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos", help="where to save the recorded trajectories")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes to use to help parallelize the trajectory replay process. This uses CPU multiprocessing and only works with the CPU simulation backend at the moment.")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for the environment.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Random seed for the environment.")

    return parser.parse_args()
'''这部分通过 argparse 模块解析命令行传入的参数，支持很多控制程序执行的选项，例如环境设置、轨迹数量、是否保存视频等。每个参数的功能如下：

--env-id：指定任务环境的ID（例如 PickCube-v1）。
--obs-mode：指定观察模式，控制环境返回的观测信息类型（例如 rgb 表示图像信息）。
--num-traj：指定要生成的轨迹数量。
--only-count-success：如果指定，只有当任务成功时才保存轨迹和视频。
--reward-mode：指定奖励模式，默认为 "dense"。
--sim-backend：指定仿真后台，可能的选项包括 auto（自动选择）、cpu 或 gpu。
--render-mode：指定渲染模式，控制如何渲染环境（例如 rgb_array 表示渲染为RGB数组图像）。
--vis：是否启用可视化（显示图形界面）。
--save-video：是否保存视频。
--traj-name：指定轨迹文件的名称。
--shader：指定渲染时使用的着色器类型，影响渲染效果。
--record-dir：指定保存轨迹和视频的目录。
--num-procs：并行处理时的进程数，主要用于回放轨迹时的并行。
--random_seed：设置环境随机种子，以确保实验可复现。
--pretrained_path：指定预训练模型的路径。'''
task2lang = {
    "PegInsertionSide-v1": "Pick up a orange-white peg and insert the orange end into the box with a hole in it.",
    "PickCube-v1": "Grasp a red cube and move it to a target goal position.",
    "StackCube-v1":  "Pick up a red cube and stack it on top of a green cube and let go of the cube without it falling.",
    "PlugCharger-v1": "Pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.",
    "PushCube-v1": "Push and move a cube to a goal region in front of it."
}
'''"zidian
PickCube-v1" 是任务名，表示一个环境或任务，可能是在某个机器人控制环境中定义的任务 ID。
"Grasp a red cube and move it to a target goal position." 是对应的任务描述，指示机器人要执行的具体动作：抓住一个红色的立方体，并将其移动到目标位置。'''
import random
import os

args = parse_args()
seed = args.random_seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

env_id = args.env_id
env = gym.make(
    env_id,
    obs_mode=args.obs_mode,
    control_mode="pd_joint_pos",
    render_mode=args.render_mode,
    reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    sensor_configs=dict(shader_pack=args.shader),
    human_render_camera_configs=dict(shader_pack=args.shader),
    viewer_camera_configs=dict(shader_pack=args.shader),
    sim_backend=args.sim_backend
)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
import hydra
import dill

checkpoint_path = args.pretrained_path
print(f"Loading policy from {checkpoint_path}. Task is {task2lang[env_id]}")

def get_policy(output_dir, device):
    
    # load checkpoint
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

policy = get_policy('./', device = 'cuda')
MAX_EPISODE_STEPS = 400
total_episodes = args.num_traj 
success_count = 0 
base_seed = 20241201
instr = task2lang[env_id]
import tqdm

DATA_STAT = {'state_min': [-0.7463043928146362, -0.0801204964518547, -0.4976441562175751, -2.657780647277832, -0.5742632150650024, 1.8309762477874756, -2.2423808574676514, 0.0, 0.0], 'state_max': [0.7645499110221863, 1.4967026710510254, 0.4650936424732208, -0.3866899907588959, 0.5505855679512024, 3.2900545597076416, 2.5737812519073486, 0.03999999910593033, 0.03999999910593033], 'action_min': [-0.7472005486488342, -0.08631071448326111, -0.4995281398296356, -2.658363103866577, -0.5751323103904724, 1.8290787935256958, -2.245187997817993, -1.0], 'action_max': [0.7654682397842407, 1.4984270334243774, 0.46786263585090637, -0.38181185722351074, 0.5517147779464722, 3.291581630706787, 2.575840711593628, 1.0], 'action_std': [0.2199309915304184, 0.18780815601348877, 0.13044124841690063, 0.30669933557510376, 0.1340624988079071, 0.24968451261520386, 0.9589747190475464, 0.9827960729598999], 'action_mean': [-0.00885344110429287, 0.5523102879524231, -0.007564723491668701, -2.0108158588409424, 0.004714342765510082, 2.615924596786499, 0.08461848646402359, -0.19301606714725494]}

state_min = torch.tensor(DATA_STAT['state_min']).cuda()
state_max = torch.tensor(DATA_STAT['state_max']).cuda()
action_min = torch.tensor(DATA_STAT['action_min']).cuda()
action_max = torch.tensor(DATA_STAT['action_max']).cuda()

for episode in tqdm.trange(total_episodes):
    obs_window = deque(maxlen=2)
    obs, _ = env.reset(seed = episode + base_seed)

    img = env.render().cuda().float()
    proprio = obs['agent']['qpos'][:].cuda()
    proprio = (proprio - state_min) / (state_max - state_min) * 2 - 1
    obs_window.append({
        'agent_pos': proprio,
        "head_cam": img.permute(0, 3, 1, 2),
    })
    obs_window.append({
        'agent_pos': proprio,
        "head_cam": img.permute(0, 3, 1, 2),
    }) 
    
    global_steps = 0
    video_frames = []

    success_time = 0
    done = False

    while global_steps < MAX_EPISODE_STEPS and not done:
        obs = obs_window[-1]
        actions = policy.predict_action(obs)
        actions = actions['action_pred'].squeeze(0)
        actions = (actions + 1) / 2 * (action_max - action_min) + action_min
        actions = actions.detach().cpu().numpy()
        actions = actions[:8]
        for idx in range(actions.shape[0]):
            action = actions[idx]
            obs, reward, terminated, truncated, info = env.step(action)
            img = env.render().cuda().float()
            proprio = obs['agent']['qpos'][:].cuda()
            proprio = (proprio - state_min) / (state_max - state_min) * 2 - 1
            obs_window.append({
                'agent_pos': proprio,
                "head_cam": img.permute(0, 3, 1, 2),
            }) 
            video_frames.append(env.render().squeeze(0).detach().cpu().numpy())
            global_steps += 1
            if terminated or truncated:
                assert "success" in info, sorted(info.keys())
                if info['success']:
                    done = True
                    success_count += 1
                    break 
    print(f"Trial {episode+1} finished, success: {info['success']}, steps: {global_steps}")

success_rate = success_count / total_episodes * 100
print(f"Tested {total_episodes} episodes, success rate: {success_rate:.2f}%")
log_file = f"results_dp_{checkpoint_path.split('/')[-1].split('.')[0]}.txt"
with open(log_file, 'a') as f:
    f.write(f"{args.env_id}:{seed}:{success_count}\n")

'''
all this .py：

使用 argparse 库来解析命令行输入的参数，例如任务环境、轨迹数量、是否保存视频、预训练模型路径等。通过这些参数，你可以控制实验的配置和运行方式。
初始化模拟环境：

通过 gym.make 初始化一个特定的模拟环境（如 PickCube-v1，这是一个用来控制机器人抓取并移动物体的环境）。
环境的参数包括观测模式、渲染模式、奖励模式等，这些决定了机器人如何感知环境以及如何进行动作和学习。
加载预训练模型：

通过 get_policy 函数加载一个预训练的策略模型。这些策略是通过深度强化学习训练得到的，能够指导机器人在环境中采取动作。
这个函数使用了 hydra 和 dill 来加载模型，确保模型的配置和权重可以正确恢复。
执行任务：

在每个回合（episode）中，机器人从环境中获取观测信息（例如机器人的关节位置、摄像头图像等），并根据预训练的策略选择动作。
执行动作后，机器人根据环境反馈（如奖励、是否完成任务等）更新其状态，并继续执行直到任务完成或者达到最大步数。
任务的执行过程还包括视频帧的录制和保存，用于后续的可视化和分析。
计算任务成功率：

每个回合结束时，代码会记录任务是否成功完成（例如机器人是否成功抓取并移动了物体）。通过统计成功的回合数，计算成功率，并将结果保存到日志文件中。
'''