# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL for locomotion tasks."""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="CustomHumanoid-Velocity-Flat-Play-v0", help="Name of the task.")
parser.add_argument("--free_cam", action="store_true", default=False, help="Disable automatic lock-on so mouse can freely pan/orbit without snapping back.")
parser.add_argument("--camera_eye", nargs=3, type=float, default=[-2.8, -2.4, 1.3], help="Camera eye position (x, y, z).")
parser.add_argument("--camera_target", nargs=3, type=float, default=[0.5, 0.0, 0.1], help="Camera target/lookat position (x, y, z).")
parser.add_argument("--env_index", type=int, default=1, help="Environment index (1-based, matching IsaacLab UI).")
parser.add_argument("--motion_file", type=str, default=None, help="Path to motion file for tracking tasks.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Default to performance rendering mode to eliminate RTX Ray-Tracing shader stalls
if args_cli.rendering_mode is None:
    args_cli.rendering_mode = "performance"
    args_cli.rendering_mode_explicit = True

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import whole_body_tracking.tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # Support motion file override for tracking tasks
    if hasattr(args_cli, "motion_file") and args_cli.motion_file is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
            env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Configure viewer camera according to recording requirements:
    # Environment Index: 1, Follow Mode: Robot, Camera Eye: (-2.8, -2.4, 1.3), Camera Target: (0.5, 0.0, 0.1)
    if hasattr(env_cfg, "viewer"):
        if args_cli.free_cam:
            # Free camera mode: static origin, mouse orbit/pan will never snap back
            env_cfg.viewer.origin_type = "env"
            env_cfg.viewer.eye = (2.0, 1.8, 0.6)
            env_cfg.viewer.lookat = (0.0, 0.0, 0.3)
        else:
            # Follow mode: follows robot (Follow Mode: Robot)
            env_cfg.viewer.origin_type = "asset_root"
            env_cfg.viewer.asset_name = "robot"
            env_cfg.viewer.env_index = max(0, args_cli.env_index - 1)
            env_cfg.viewer.eye = tuple(args_cli.camera_eye)
            env_cfg.viewer.lookat = tuple(args_cli.camera_target)
            print(f"[INFO] Viewer Camera Configured: Follow={env_cfg.viewer.asset_name}, Eye={env_cfg.viewer.eye}, Target={env_cfg.viewer.lookat}, EnvIndex={args_cli.env_index}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    log_dir = os.path.dirname(resume_path)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    # obtain policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    print("[INFO] ========================================================")
    print("[INFO] [SUCCESS] Simulation GUI is ready and active!")
    print("[INFO] Robot policy inference running in 3D viewport...")
    print("[INFO] Use Left Mouse Button to Rotate, Middle to Pan, Scroll to Zoom.")
    print("[INFO] Press Ctrl+C in this terminal to exit.")
    print("[INFO] ========================================================")

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        timestep += 1
        if timestep % 100 == 0:
            print(f"[INFO] Simulated {timestep} steps (realtime play active)...")

        if args_cli.video and timestep >= args_cli.video_length:
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    import os, sys
    sys.stdout.flush()
    os._exit(0)
