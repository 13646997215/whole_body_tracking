import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl"))

from isaaclab.app import AppLauncher
import cli_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="CustomHumanoid-Tracking-Flat-Play-v0")
parser.add_argument("--num_steps", type=int, default=300)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(checkpoint="exported_models/custom_humanoid/custom_humanoid_tracking_final.pt")
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
def run_telemetry(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = 1
    
    print(f"\n[TRACKING DIAGNOSTIC] Loading task: {args_cli.task}")
    print(f"[TRACKING DIAGNOSTIC] Checkpoint: {args_cli.checkpoint}")
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_term("motion")
    
    joint_names = robot.joint_names
    upper_body_joints = [j for j in joint_names if any(w in j for w in ["shoulder", "elbow", "wrist"])]
    lower_body_joints = [j for j in joint_names if any(w in j for w in ["hip", "knee", "ankle"])]
    upper_idx = [joint_names.index(j) for j in upper_body_joints]
    lower_idx = [joint_names.index(j) for j in lower_body_joints]
    
    all_errors = []
    upper_errors = []
    lower_errors = []
    dones_count = 0
    
    print(f"\n[TELEMETRY START] BeyondMimic Dance Tracking Trajectory ({args_cli.num_steps} steps):")
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            
        ref_q = command.joint_pos[0].cpu().numpy()
        act_q = command.robot_joint_pos[0].cpu().numpy()
        err_deg = np.abs(ref_q - act_q) * 180.0 / np.pi
        
        all_errors.append(err_deg)
        upper_errors.append(err_deg[upper_idx])
        lower_errors.append(err_deg[lower_idx])
        
        if step > 0 and dones[0].item():
            dones_count += 1
            
        if step % 50 == 0 or step == args_cli.num_steps - 1:
            t = step * 0.02
            print(f"Time {t:5.2f}s (Step {step:3d}): Overall RMSE={np.sqrt(np.mean(err_deg**2)):5.2f} deg, Upper RMSE={np.sqrt(np.mean(err_deg[upper_idx]**2)):5.2f} deg, Lower RMSE={np.sqrt(np.mean(err_deg[lower_idx]**2)):5.2f} deg, done={dones[0].item()}")
            
    all_errors = np.array(all_errors)
    upper_errors = np.array(upper_errors)
    lower_errors = np.array(lower_errors)
    
    print(f"\n[TRACKING TELEMETRY SUMMARY]")
    print(f"Total Steps Evaluated: {args_cli.num_steps} ({args_cli.num_steps * 0.02:.1f}s)")
    print(f"Survival Rate: {(1.0 - dones_count / max(1, args_cli.num_steps)) * 100:.1f}% (Falls / Collapses: {dones_count})")
    print(f"Overall Joint Tracking RMSE: {np.sqrt(np.mean(all_errors**2)):.2f} deg (Mean Abs Error: {np.mean(all_errors):.2f} deg)")
    print(f"Upper Body Expressive Tracking RMSE: {np.sqrt(np.mean(upper_errors**2)):.2f} deg (Mean Abs Error: {np.mean(upper_errors):.2f} deg)")
    print(f"Lower Body Stance Tracking RMSE: {np.sqrt(np.mean(lower_errors**2)):.2f} deg (Mean Abs Error: {np.mean(lower_errors):.2f} deg)")
    
    pass_criteria = dones_count == 0 and np.sqrt(np.mean(all_errors**2)) < 15.0
    if pass_criteria:
        print(f"[VERDICT] PASS: Fluid, high-fidelity expressive dance tracking with 100% survival!")
    else:
        print(f"[VERDICT] REVIEW NEEDED: Falls={dones_count}, RMSE={np.sqrt(np.mean(all_errors**2)):.2f} deg")
        
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    run_telemetry()
