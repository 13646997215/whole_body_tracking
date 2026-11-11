# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for fall recovery tasks."""

import math
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz


def reset_diverse_fall_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Multi-tier curriculum reset for robust fall recovery and motionless standing:
    - Tier 1 (30%): Crouch / Stumble recovery (near upright, root z~0.52m, tilt ~20 deg)
    - Tier 2 (30%): Kneeling / Sitting recovery (root z~0.36m, tilt ~40 deg)
    - Tier 3 (40%): Full Ground Lying Recovery (supine, prone, lateral left, lateral right, root z~0.20m)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    # When evaluating a single environment (play/diagnostic) or 70% in training, prioritize flat ground recovery
    if num_resets == 1:
        selector = torch.ones(num_resets, device=asset.device)  # 100% flat ground
    else:
        selector = torch.rand(num_resets, device=asset.device)

    roll = torch.zeros(num_resets, device=asset.device)
    pitch = torch.zeros(num_resets, device=asset.device)
    yaw = torch.empty(num_resets, device=asset.device).uniform_(-math.pi, math.pi)
    root_z = torch.zeros(num_resets, device=asset.device)

    # Default joint pos reference
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    # Tier 1: Crouch / Stumbled Recovery (15%)
    mask_crouch = selector < 0.15
    n_crouch = mask_crouch.sum().item()
    if n_crouch > 0:
        roll[mask_crouch] = torch.empty(n_crouch, device=asset.device).uniform_(-0.25, 0.25)
        pitch[mask_crouch] = torch.empty(n_crouch, device=asset.device).uniform_(-0.35, 0.35)
        root_z[mask_crouch] = torch.empty(n_crouch, device=asset.device).uniform_(0.48, 0.56)

    # Tier 2: Kneeling / Sitting Recovery (15%)
    mask_kneel = (selector >= 0.15) & (selector < 0.30)
    n_kneel = mask_kneel.sum().item()
    if n_kneel > 0:
        roll[mask_kneel] = torch.empty(n_kneel, device=asset.device).uniform_(-0.40, 0.40)
        pitch[mask_kneel] = torch.empty(n_kneel, device=asset.device).uniform_(-0.70, 0.70)
        root_z[mask_kneel] = torch.empty(n_kneel, device=asset.device).uniform_(0.32, 0.40)

    # Tier 3: Full Ground Lying Recovery (70% in training, 100% in single env play/eval)
    mask_ground = selector >= 0.30
    n_ground = mask_ground.sum().item()
    if n_ground > 0:
        ground_types = torch.randint(0, 4, (n_ground,), device=asset.device)
        ground_roll = torch.zeros(n_ground, device=asset.device)
        ground_pitch = torch.zeros(n_ground, device=asset.device)

        # 0: Supine (仰卧)
        m_sup = ground_types == 0
        ground_pitch[m_sup] = -math.pi / 2.0 + torch.randn(m_sup.sum(), device=asset.device) * 0.1
        ground_roll[m_sup] = torch.randn(m_sup.sum(), device=asset.device) * 0.1

        # 1: Prone (俯卧)
        m_pro = ground_types == 1
        ground_pitch[m_pro] = math.pi / 2.0 + torch.randn(m_pro.sum(), device=asset.device) * 0.1
        ground_roll[m_pro] = torch.randn(m_pro.sum(), device=asset.device) * 0.1

        # 2: Lateral Left (左侧卧)
        m_left = ground_types == 2
        ground_roll[m_left] = -math.pi / 2.0 + torch.randn(m_left.sum(), device=asset.device) * 0.1
        ground_pitch[m_left] = torch.randn(m_left.sum(), device=asset.device) * 0.1

        # 3: Lateral Right (右侧卧)
        m_right = ground_types == 3
        ground_roll[m_right] = math.pi / 2.0 + torch.randn(m_right.sum(), device=asset.device) * 0.1
        ground_pitch[m_right] = torch.randn(m_right.sum(), device=asset.device) * 0.1

        roll[mask_ground] = ground_roll
        pitch[mask_ground] = ground_pitch
        root_z[mask_ground] = 0.20 + torch.empty(n_ground, device=asset.device).uniform_(-0.02, 0.02)

    # Convert Euler angles to quaternion
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    # Get root states
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, 0] += torch.empty(num_resets, device=asset.device).uniform_(-0.3, 0.3)
    root_states[:, 1] += torch.empty(num_resets, device=asset.device).uniform_(-0.3, 0.3)
    root_states[:, 2] = root_z
    root_states[:, 3:7] = quat
    root_states[:, 7:13] = 0.0  # Zero initial velocities

    # Write root state to simulation
    asset.write_root_state_to_sim(root_states, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
