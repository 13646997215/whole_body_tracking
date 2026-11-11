import os
import sys
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "rsl_rl"))

from isaaclab.app import AppLauncher
import cli_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="CustomHumanoid-FallRecovery-v0")
parser.add_argument("--num_steps", type=int, default=150)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(checkpoint="exported_models/custom_humanoid/custom_humanoid_fall_recovery_final.pt")
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
    
    print(f"\n[FALL RECOVERY DIAGNOSTIC] Loading task: {args_cli.task}")
    print(f"[FALL RECOVERY DIAGNOSTIC] Checkpoint: {args_cli.checkpoint}")
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    head_id = robot.body_names.index("head_roll_link")
    
    head_z_list = []
    root_z_list = []
    lin_vel_list = []
    ang_vel_list = []
    j_vel_list = []

    print(f"\n[TELEMETRY START] Fall Recovery Trajectory:")
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        head_pos = robot.data.body_pos_w[0, head_id].cpu().numpy()
        grav = robot.data.projected_gravity_b[0].cpu().numpy()
        lin_v = torch.norm(robot.data.root_lin_vel_w[0]).item()
        ang_v = torch.norm(robot.data.root_ang_vel_w[0]).item()
        j_v = torch.norm(robot.data.joint_vel[0]).item()

        head_z_list.append(head_pos[2])
        root_z_list.append(root_pos[2])
        lin_vel_list.append(lin_v)
        ang_vel_list.append(ang_v)
        j_vel_list.append(j_v)

        if step % 20 == 0 or step == args_cli.num_steps - 1:
            t = step * 0.02
            lk_id = robot.joint_names.index("left_knee_joint")
            rk_id = robot.joint_names.index("right_knee_joint")
            lhp_id = robot.joint_names.index("left_hip_pitch_joint")
            rhp_id = robot.joint_names.index("right_hip_pitch_joint")
            q_knee = (robot.data.joint_pos[0, lk_id].item(), robot.data.joint_pos[0, rk_id].item())
            q_hip = (robot.data.joint_pos[0, lhp_id].item(), robot.data.joint_pos[0, rhp_id].item())
            if step == args_cli.num_steps - 1:
                print("\n[ALL 26 JOINT ANGLES AT FINAL STEP]:")
                for jname in robot.joint_names:
                    jid = robot.joint_names.index(jname)
                    val = robot.data.joint_pos[0, jid].item()
                    dval = robot.data.default_joint_pos[0, jid].item()
                    print(f"  {jname:28s}: {val:7.3f} rad ({val*180/3.14159:6.1f} deg) [default: {dval:7.3f}]")

    standing_indices = [i for i, hz in enumerate(head_z_list) if hz >= 1.05]
    if standing_indices:
        first_stand_step = standing_indices[0]
        settle_step = min(first_stand_step + 20, args_cli.num_steps - 20)
        standing_steps = list(range(settle_step, args_cli.num_steps))
        rise_time = first_stand_step * 0.02
        print(f"\n[RISE EVENT] Upright standing height reached at step {first_stand_step} (Time: {rise_time:.2f}s, Head_Z: {head_z_list[first_stand_step]:.3f}m)!")
    else:
        settle_step = int(args_cli.num_steps * 0.7)
        standing_steps = list(range(settle_step, args_cli.num_steps))
        print(f"\n[RISE EVENT] Warning: Head did not reach >= 1.05m during the trajectory.")

    standing_head_z = [head_z_list[i] for i in standing_steps]
    standing_root_z = [root_z_list[i] for i in standing_steps]
    standing_lin_v = [lin_vel_list[i] for i in standing_steps]
    standing_ang_v = [ang_vel_list[i] for i in standing_steps]

    print(f"\n[FALL RECOVERY TELEMETRY SUMMARY]")
    print(f"Initial Head Z: {head_z_list[0]:.3f}m -> Max Head Z: {max(head_z_list):.3f}m -> Final Head Z: {head_z_list[-1]:.3f}m")
    print(f"Initial Root Z: {root_z_list[0]:.3f}m -> Max Root Z: {max(root_z_list):.3f}m -> Final Root Z: {root_z_list[-1]:.3f}m")
    print(f"Sustained Motionless Standing Phase ({settle_step*0.02:.2f}s - {args_cli.num_steps*0.02:.2f}s, Steps {settle_step}-{args_cli.num_steps}):")
    print(f"  Head Z: mean={np.mean(standing_head_z):.3f}m, min={min(standing_head_z):.3f}m, max={max(standing_head_z):.3f}m")
    print(f"  Root Z: mean={np.mean(standing_root_z):.3f}m, min={min(standing_root_z):.3f}m, max={max(standing_root_z):.3f}m")
    print(f"  Mean LinVel: {np.mean(standing_lin_v):.3f}m/s, Max LinVel: {max(standing_lin_v):.3f}m/s")
    print(f"  Mean AngVel: {np.mean(standing_ang_v):.3f}rad/s, Max AngVel: {max(standing_ang_v):.3f}rad/s")

    is_standing = min(standing_head_z) >= 1.05 and min(standing_root_z) >= 0.50
    is_motionless = np.mean(standing_lin_v) < 0.25 and np.mean(standing_ang_v) < 0.50

    if is_standing and is_motionless:
        print(f"[VERDICT] PASS: Rock-solid motionless standing achieved! (Head: {np.mean(standing_head_z):.3f}m, LinVel: {np.mean(standing_lin_v):.3f}m/s, AngVel: {np.mean(standing_ang_v):.3f}rad/s)")
    elif is_standing:
        print(f"[VERDICT] PARTIAL PASS: Standing upright (Head: {np.mean(standing_head_z):.3f}m) but still balancing/damping (LinVel: {np.mean(standing_lin_v):.3f}m/s)")
    else:
        print(f"[VERDICT] FAIL: Head dropped below standing height (min={min(standing_head_z):.3f}m < 1.05m)")

    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    run_telemetry()
