import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "rsl_rl"))

from isaaclab.app import AppLauncher
import cli_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="CustomHumanoid-Velocity-Stairs-Play-v0")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def run_debug(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = 1
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint)) if os.path.exists(args_cli.checkpoint) else os.path.join("logs", "rsl_rl", "custom_humanoid_stairs", args_cli.load_run)
    ckpt_path = os.path.abspath(args_cli.checkpoint) if os.path.exists(args_cli.checkpoint) else os.path.join(log_dir, args_cli.checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(ckpt_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    foot_indices, _ = robot.find_bodies(".*_ankle_roll_link")
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    sensor_foot_ids = [contact_sensor.body_names.index("left_ankle_roll_link"), contact_sensor.body_names.index("right_ankle_roll_link")]
    
    print("\n--- Detailed Step Analysis [80 - 125] ---")
    for step in range(130):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rews, dones, _ = env.step(actions)
            
        if step >= 80 and step <= 125:
            pos = robot.data.root_pos_w[0].cpu().numpy()
            vel = robot.data.root_lin_vel_w[0].cpu().numpy()
            grav = robot.data.projected_gravity_b[0].cpu().numpy()
            feet_z = robot.data.body_pos_w[0, foot_indices, 2].cpu().numpy()
            feet_x = robot.data.body_pos_w[0, foot_indices, 0].cpu().numpy()
            knee_q = robot.data.joint_pos[0, [robot.joint_names.index("left_knee_joint"), robot.joint_names.index("right_knee_joint")]].cpu().numpy()
            cf = contact_sensor.data.net_forces_w_history[0, :, sensor_foot_ids, :].norm(dim=-1).max(dim=0)[0].cpu().numpy()
            
            # Print detailed line
            print(f"Step {step:3d} | X={pos[0]:.2f} Z={pos[2]:.2f} Vx={vel[0]:.2f} Vz={vel[2]:.2f} | gx={grav[0]:.2f} | FootX=({feet_x[0]:.2f}, {feet_x[1]:.2f}) FootZ=({feet_z[0]:.2f}, {feet_z[1]:.2f}) | CF=({cf[0]:.0f}, {cf[1]:.0f}) | Rew={rews[0].item():.2f} Done={dones[0].item()}")
            
    os._exit(0)

if __name__ == "__main__":
    run_debug()
