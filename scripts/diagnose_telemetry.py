import os
import sys
import torch
import numpy as np

# append rsl_rl
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "rsl_rl"))

from isaaclab.app import AppLauncher
import cli_args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=150)
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
def run_telemetry(env_cfg, agent_cfg):
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = 1
    
    print(f"\n[DIAGNOSTIC] Loading task: {args_cli.task}")
    print(f"[DIAGNOSTIC] Checkpoint: {args_cli.checkpoint}")
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    obs, _ = env.reset()
    
    robot = env.unwrapped.scene["robot"]
    
    foot_indices, _ = robot.find_bodies(".*_ankle_roll_link")
    
    print(f"\n[TELEMETRY START] Initial State:")
    root_pos = robot.data.root_pos_w[0].cpu().numpy()
    root_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(f"Initial Pos: X={root_pos[0]:.3f}, Y={root_pos[1]:.3f}, Z={root_pos[2]:.3f}")
    
    z_history = []
    x_history = []
    y_history = []
    pitch_history = []
    feet_z_history = []
    knee_indices = [robot.joint_names.index("left_knee_joint"), robot.joint_names.index("right_knee_joint")]
    max_knee_flexion = [0.0, 0.0]
    double_flight_steps = 0
    single_stance_steps = 0
    
    contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
    sensor_foot_ids = [contact_sensor.body_names.index("left_ankle_roll_link"), contact_sensor.body_names.index("right_ankle_roll_link")]
    
    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            
        pos = robot.data.root_pos_w[0].cpu().numpy()
        grav = robot.data.projected_gravity_b[0].cpu().numpy()
        feet_z = robot.data.body_pos_w[0, foot_indices, 2].cpu().numpy()
        feet_x = robot.data.body_pos_w[0, foot_indices, 0].cpu().numpy()
        knee_q = robot.data.joint_pos[0, knee_indices].cpu().numpy()
        max_knee_flexion[0] = max(max_knee_flexion[0], float(knee_q[0]))
        max_knee_flexion[1] = max(max_knee_flexion[1], float(knee_q[1]))

        # Check foot contacts
        net_forces = contact_sensor.data.net_forces_w_history[0, :, sensor_foot_ids, :].norm(dim=-1).max(dim=0)[0]
        has_c0 = net_forces[0].item() > 1.0
        has_c1 = net_forces[1].item() > 1.0
        if not has_c0 and not has_c1:
            double_flight_steps += 1
        elif (has_c0 and not has_c1) or (has_c1 and not has_c0):
            single_stance_steps += 1
        
        z_history.append(pos[2])
        x_history.append(pos[0])
        y_history.append(pos[1])
        pitch_history.append(grav[0])
        feet_z_history.append(feet_z)
        
        if dones[0].item() or step % 20 == 0 or step == args_cli.num_steps - 1:
            # Check contact forces on all bodies
            all_cf = env.unwrapped.scene.sensors["contact_forces"]
            cf_m = all_cf.data.net_forces_w[0].norm(dim=-1)
            high_cf = [(all_cf.body_names[i], round(cf_m[i].item(), 1)) for i in range(len(all_cf.body_names)) if cf_m[i].item() > 1.0]
            print(f"Step {step:3d}: Pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), FootX=({feet_x[0]:.2f}, {feet_x[1]:.2f}), FootZ=({feet_z[0]:.3f}, {feet_z[1]:.3f}), Knees=({knee_q[0]:.2f}, {knee_q[1]:.2f})rad, done={dones[0].item()}, contacts={high_cf}")
            
    feet_z_arr = np.array(feet_z_history)
    print(f"\n[TELEMETRY SUMMARY]")
    print(f"X delta: {x_history[-1] - x_history[0]:.3f} m")
    print(f"Y delta: {y_history[-1] - y_history[0]:.3f} m")
    print(f"Root Z: initial={z_history[0]:.3f} m, final={z_history[-1]:.3f} m, min={min(z_history):.3f} m, max={max(z_history):.3f} m, delta={z_history[-1] - z_history[0]:.3f} m")
    print(f"Feet Z: min=({feet_z_arr[:,0].min():.3f}, {feet_z_arr[:,1].min():.3f}), max=({feet_z_arr[:,0].max():.3f}, {feet_z_arr[:,1].max():.3f})")
    print(f"Max Knee Flexion: Left={max_knee_flexion[0]:.2f} rad ({np.degrees(max_knee_flexion[0]):.1f} deg), Right={max_knee_flexion[1]:.2f} rad ({np.degrees(max_knee_flexion[1]):.1f} deg)")
    print(f"Gait Phases: Single-Stance={single_stance_steps}/{args_cli.num_steps} ({single_stance_steps/args_cli.num_steps*100:.1f}%), Bunny-Hopping Double-Flight={double_flight_steps}/{args_cli.num_steps} ({double_flight_steps/args_cli.num_steps*100:.1f}%)")
    print(f"Avg Pitch gx: {np.mean(pitch_history):.3f}")
    
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    run_telemetry()
