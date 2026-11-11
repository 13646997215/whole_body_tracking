# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for fall recovery tasks."""

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse


def head_height_range(
    env,
    min_nominal: float = 1.12,
    max_nominal: float = 1.26,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Target bounded range [1.12m, 1.26m] for head height (user requirement: range, not just lower bound):
    - Continuous rising progress when head < 1.12m
    - Peak plateau reward inside [1.12m, 1.26m]
    - Severe quadratic penalty above 1.26m to eliminate jumping/hopping bug
    """
    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    progress = torch.clamp((head_z - 0.20) / (min_nominal - 0.20), min=0.0, max=1.0)
    in_range = ((head_z >= min_nominal) & (head_z <= max_nominal)).float()
    overshoot = torch.clamp(head_z - max_nominal, min=0.0)
    jump_penalty = 20.0 * torch.square(overshoot)
    return progress + 6.0 * in_range - jump_penalty


def root_height_range(
    env,
    min_nominal: float = 0.58,
    max_nominal: float = 0.70,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Target bounded range [0.58m, 0.70m] for pelvis height:
    - Continuous rising progress when root < 0.58m
    - Peak plateau reward inside [0.58m, 0.70m]
    - Severe quadratic penalty above 0.70m to eliminate leaping
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    progress = torch.clamp((root_z - 0.18) / (min_nominal - 0.18), min=0.0, max=1.0)
    in_range = ((root_z >= min_nominal) & (root_z <= max_nominal)).float()
    overshoot = torch.clamp(root_z - max_nominal, min=0.0)
    jump_penalty = 20.0 * torch.square(overshoot)
    return progress + 5.0 * in_range - jump_penalty


def com_support_alignment(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Reward keeping root COM horizontally centered directly between both feet.

    Provides the fundamental stabilizing physics for rock-solid stationary standing without tipping!
    """
    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    root_xy = asset.data.root_pos_w[:, 0:2]
    feet_xy = asset.data.body_pos_w[:, feet_cfg.body_ids, 0:2]
    mid_feet_xy = torch.mean(feet_xy, dim=1)

    horizontal_offset_sq = torch.sum(torch.square(root_xy - mid_feet_xy), dim=-1)
    standing_mask = torch.clamp((head_z - 0.60) / 0.30, min=0.0, max=1.0)
    return standing_mask * torch.exp(-15.0 * horizontal_offset_sq)


def upright_posture_bonus(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Reward keeping torso aligned with world vertical, active once head lifts above 0.45m."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=asset.device).repeat(env.num_envs, 1)
    proj_grav = quat_rotate_inverse(asset.data.root_quat_w, gravity_vec)
    upright = torch.clamp((-proj_grav[:, 2] + 1.0) / 2.0, min=0.0, max=1.0)

    head_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
    height_gate = torch.clamp((head_z - 0.45) / (1.00 - 0.45), min=0.0, max=1.0)
    return (upright**2) * height_gate


def excessive_vertical_vel_penalty(
    env,
    max_vz: float = 0.50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize excessive upward vertical velocity to eliminate violent explosive leaping."""
    asset: Articulation = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    excess = torch.clamp(vz - max_vz, min=0.0)
    return torch.square(excess)


def root_ang_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize spinning and tumbling during recovery to maintain stable torso orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)


def upper_body_ground_contact_penalty(
    env,
    min_head_z: float = 0.70,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base_link", "torso_link"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Penalize torso and pelvis touching the floor once elevated (>0.70m) or after grace period (>2.0s)."""
    asset: Articulation = env.scene[head_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    step_time = env.episode_length_buf * env.step_dt
    active_mask = ((head_z > min_head_z) | (step_time > 2.0)).float()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    has_contact = (contacts.sum(dim=-1) > 0).float()
    return active_mask * has_contact


def feet_contact_when_standing(
    env,
    min_head_height: float = 0.85,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Reward dual-foot support contact on ground once head is elevated to standing zone (>= 0.85m)."""
    asset: Articulation = env.scene[head_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    standing_mask = torch.clamp((head_z - min_head_height) / 0.15, min=0.0, max=1.0)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 5.0).float()
    both_feet = (contacts[:, 0] > 0.0) & (contacts[:, 1] > 0.0)
    return standing_mask * (contacts.sum(dim=1) * 0.5 + both_feet.float() * 2.0)


def stand_still_bonus(
    env,
    min_head_z: float = 1.00,
    max_head_z: float = 1.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Substantial reward for zero linear, angular, and joint velocities once standing in range."""
    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]

    standing_mask = torch.clamp((head_z - min_head_z) / 0.08, min=0.0, max=1.0)
    not_jumping = (head_z <= max_head_z).float()
    mask = standing_mask * not_jumping

    lin_vel_sq = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)
    ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel), dim=1)

    # Posture regularization towards nominal symmetrical standing pose
    joint_pos_diff = asset.data.joint_pos - asset.data.default_joint_pos
    posture_error = torch.mean(torch.square(joint_pos_diff), dim=1)

    # Smooth broad-kernel terms to provide non-vanishing gradients across all speeds
    lin_score = torch.exp(-2.5 * lin_vel_sq)
    ang_score = torch.exp(-0.8 * ang_vel_sq)
    joint_score = torch.exp(-0.5 * joint_vel_sq)
    posture_score = torch.exp(-2.5 * posture_error)

    vel_score = 0.4 * lin_score + 0.3 * ang_score + 0.3 * joint_score

    return mask * (2.0 * vel_score + 1.5 * posture_score)


def airborne_jumping_penalty(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strictly penalize both feet airborne once pelvis is lifted off the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    elevated_mask = (root_z > 0.45).float()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    no_feet_contact = (contacts.sum(dim=1) == 0).float()
    return elevated_mask * no_feet_contact


def fall_after_recovery(
    env,
    grace_time_s: float = 3.2,
    min_head_z: float = 0.88,
    min_root_z: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["base_link", "torso_link", "head_roll_link"]),
) -> torch.Tensor:
    """Terminate if humanoid collapses, tilts over, or falls after initial recovery grace period (3.2s).

    Strictly eliminates the yo-yo cycle of falling and standing back up.
    """
    step_time = env.episode_length_buf * env.step_dt
    past_grace = step_time > grace_time_s

    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    root_z = asset.data.root_pos_w[:, 2]

    # Height collapse detection
    height_collapse = (head_z < min_head_z) | (root_z < min_root_z)

    # Torso, pelvis, head contact detection
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    upper_contact = contacts.any(dim=-1)

    return past_grace & (height_collapse | upper_contact)


def standing_still_lin_vel_penalty(
    env,
    min_head_z: float = 1.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Strict penalty on horizontal drift (vx, vy) and downward collapse once standing.
    Allows upward velocity during the final standing phase to avoid trapping robot in a crouch.
    """
    asset: Articulation = env.scene[head_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    mask = torch.clamp((head_z - min_head_z) / 0.08, min=0.0, max=1.0)
    # Penalize horizontal drift (vx, vy)
    h_vel_sq = torch.square(asset.data.root_lin_vel_w[:, 0]) + torch.square(asset.data.root_lin_vel_w[:, 1])
    # Penalize downward velocity (vz < 0)
    vz_down = torch.square(torch.clamp(-asset.data.root_lin_vel_w[:, 2], min=0.0))
    return mask * (h_vel_sq + vz_down)


def standing_still_ang_vel_penalty(
    env,
    min_head_z: float = 1.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Strict penalty on angular velocity once standing to eliminate torso wobbling/twisting."""
    asset: Articulation = env.scene[head_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    mask = torch.clamp((head_z - min_head_z) / 0.08, min=0.0, max=1.0)
    ang_vel_sq = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    return mask * ang_vel_sq


def standing_knee_extension_progress(
    env,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
    left_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_knee_joint"]),
    right_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["right_knee_joint"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward extending knees from crouch/kneeling (standing upright) once upper body is off ground (head_z >= 0.65m).
    Directly pulls robot out of the kneeling/squat cheat and drives it into full upright standing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    rise_mask = torch.clamp((head_z - 0.65) / 0.35, min=0.0, max=1.0)
    q_lk = asset.data.joint_pos[:, left_knee_cfg.joint_ids[0]]
    q_rk = asset.data.joint_pos[:, right_knee_cfg.joint_ids[0]]
    # Reward knee angles close to straight (nominal 0.15 ~ 0.25 rad)
    ext_lk = torch.exp(-5.0 * torch.square(q_lk - 0.20))
    ext_rk = torch.exp(-5.0 * torch.square(q_rk - 0.20))
    return rise_mask * 0.5 * (ext_lk + ext_rk)


def standing_still_joint_vel_penalty(
    env,
    min_head_z: float = 0.95,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Strict penalty on joint velocity once standing to freeze completely still without fidgeting."""
    asset: Articulation = env.scene[head_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    mask = torch.clamp((head_z - min_head_z) / 0.10, min=0.0, max=1.0)
    joint_vel_sq = torch.mean(torch.square(asset.data.joint_vel), dim=1)
    return mask * joint_vel_sq


def standing_feet_alignment_penalty(
    env,
    min_head_z: float = 0.90,
    max_split_x: float = 0.08,
    target_width_y: float = 0.20,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Strictly penalize feet being split apart in X (lunge / 弓步 / 劈叉) or too wide/narrow in Y when standing.
    Forces both feet to be positioned side-by-side in a clean, natural upright standing posture.
    """
    asset: Articulation = env.scene[head_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    standing_mask = torch.clamp((head_z - min_head_z) / 0.15, min=0.0, max=1.0)

    feet_pos = asset.data.body_pos_w[:, feet_cfg.body_ids, :]
    # Longitudinal split disparity |x_0 - x_1|
    dx = torch.abs(feet_pos[:, 0, 0] - feet_pos[:, 1, 0])
    lunge_excess = torch.clamp(dx - max_split_x, min=0.0)

    # Lateral width disparity |y_0 - y_1|
    dy = torch.abs(feet_pos[:, 0, 1] - feet_pos[:, 1, 1])
    width_err = torch.square(dy - target_width_y)

    return standing_mask * (15.0 * torch.square(lunge_excess) + 5.0 * width_err)


def standing_knee_symmetry_penalty(
    env,
    min_head_z: float = 0.90,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    left_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_knee_joint"]),
    right_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["right_knee_joint"]),
) -> torch.Tensor:
    """Strictly penalize asymmetry between left and right knees and excessive knee crouching when standing.
    Forces both legs to be symmetrically upright and extended.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    standing_mask = torch.clamp((head_z - min_head_z) / 0.15, min=0.0, max=1.0)

    q_lk = asset.data.joint_pos[:, left_knee_cfg.joint_ids[0]]
    q_rk = asset.data.joint_pos[:, right_knee_cfg.joint_ids[0]]

    # Knee asymmetry penalty
    knee_diff_sq = torch.square(q_lk - q_rk)
    # Excessive crouch penalty (knees should be extended, nominal <= 0.45 rad)
    excess_lk = torch.clamp(q_lk - 0.45, min=0.0)
    excess_rk = torch.clamp(q_rk - 0.45, min=0.0)
    crouch_sq = torch.square(excess_lk) + torch.square(excess_rk)

    return standing_mask * (10.0 * knee_diff_sq + 5.0 * crouch_sq)


def hands_pushup_support_reward(
    env,
    max_head_z: float = 0.85,
    max_time_s: float = 2.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_wrist_.*"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward hands/wrists making solid contact with the floor to push the body up during initial ground recovery."""
    step_time = env.episode_length_buf * env.step_dt
    time_mask = (step_time < max_time_s).float()

    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    ground_mask = torch.clamp((max_head_z - head_z) / 0.30, min=0.0, max=1.0)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hand_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    hand_contact = (hand_forces > 5.0).float()

    vz = torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0, max=1.0)
    push_score = hand_contact.sum(dim=-1) * (1.0 + 2.0 * vz)

    return time_mask * ground_mask * push_score


def nominal_standing_posture_bonus(
    env,
    min_head_z: float = 1.05,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Substantial reward for all 26 joints converging to the clean nominal standing pose (arms at sides, straight legs)."""
    asset: Articulation = env.scene[asset_cfg.name]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]
    standing_mask = torch.clamp((head_z - min_head_z) / 0.10, min=0.0, max=1.0)

    joint_pos_diff = asset.data.joint_pos - asset.data.default_joint_pos
    posture_error = torch.mean(torch.square(joint_pos_diff), dim=1)
    return standing_mask * torch.exp(-8.0 * posture_error)


def knee_ground_contact_penalty(
    env,
    min_time_s: float = 1.8,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_knee_link"),
) -> torch.Tensor:
    """Strictly penalize knees touching the floor after the initial rising phase (t > 1.8s).
    Forces the robot to rise up off its knees and support its weight entirely on its feet!
    """
    step_time = env.episode_length_buf * env.step_dt
    active_mask = (step_time > min_time_s).float()

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    knee_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    knee_contact = (knee_forces > 1.0).any(dim=-1).float()
    return active_mask * knee_contact


def standing_knee_crouch_penalty(
    env,
    max_knee_angle: float = 0.45,
    min_time_s: float = 1.8,
    left_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_knee_joint"]),
    right_knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["right_knee_joint"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strict penalty on knees remaining bent (> 0.45 rad / 25 deg) after 1.8s.
    Forces both legs to fully straighten and extend into full upright standing.
    """
    step_time = env.episode_length_buf * env.step_dt
    active_mask = (step_time > min_time_s).float()

    asset: Articulation = env.scene[asset_cfg.name]
    q_lk = asset.data.joint_pos[:, left_knee_cfg.joint_ids[0]]
    q_rk = asset.data.joint_pos[:, right_knee_cfg.joint_ids[0]]

    excess_lk = torch.clamp(q_lk - max_knee_angle, min=0.0)
    excess_rk = torch.clamp(q_rk - max_knee_angle, min=0.0)
    return active_mask * (torch.square(excess_lk) + torch.square(excess_rk))


def standing_posture_deviation_penalty(
    env,
    min_time_s: float = 1.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strict penalty on joint deviations from nominal standing posture after initial rising (t > 1.8s).
    Forces waist, hips, legs, and arms to all return to the clean nominal upright standing pose,
    completely destroying sideways waist twists and leg splits!
    """
    step_time = env.episode_length_buf * env.step_dt
    active_mask = (step_time > min_time_s).float()

    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.joint_pos - asset.data.default_joint_pos
    error = torch.mean(torch.square(diff), dim=-1)
    return active_mask * error


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding across the floor when in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def standing_action_attenuation(
    env,
    min_root_z: float = 0.55,
    min_time_s: float = 1.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize active actions once standing (time >= 1.4s, root >= 0.55m) to allow nominal PD stiffness to lock into a motionless statue."""
    step_time = env.episode_length_buf * env.step_dt
    time_mask = torch.clamp((step_time - min_time_s) / 0.3, min=0.0, max=1.0)
    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    root_mask = torch.clamp((root_z - min_root_z) / 0.10, min=0.0, max=1.0)
    standing_mask = time_mask * root_mask
    act_sq = torch.sum(torch.square(env.action_manager.action), dim=1)
    return standing_mask * act_sq


def ankle_roll_margin_penalty(
    env,
    min_root_z: float = 0.55,
    min_time_s: float = 1.4,
    max_roll: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ankle_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"]),
) -> torch.Tensor:
    """Penalize ankle roll angle exceeding +/- 0.08 rad (4.5 deg) once standing."""
    step_time = env.episode_length_buf * env.step_dt
    time_mask = torch.clamp((step_time - min_time_s) / 0.3, min=0.0, max=1.0)
    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    root_mask = torch.clamp((root_z - min_root_z) / 0.10, min=0.0, max=1.0)
    standing_mask = time_mask * root_mask

    ankle_roll = asset.data.joint_pos[:, ankle_cfg.joint_ids]
    excess = torch.clamp(torch.abs(ankle_roll) - max_roll, min=0.0)
    return standing_mask * torch.sum(torch.square(excess), dim=-1)


def standing_torso_yaw_penalty(
    env,
    min_root_z: float = 0.55,
    min_time_s: float = 1.4,
    max_yaw: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["torso_joint"]),
    hip_yaw_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"]),
) -> torch.Tensor:
    """Penalize torso twist and hip yaw once standing to eliminate contortion and keep body strictly facing forward."""
    step_time = env.episode_length_buf * env.step_dt
    time_mask = torch.clamp((step_time - min_time_s) / 0.3, min=0.0, max=1.0)
    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    root_mask = torch.clamp((root_z - min_root_z) / 0.10, min=0.0, max=1.0)
    standing_mask = time_mask * root_mask

    torso_q = asset.data.joint_pos[:, torso_cfg.joint_ids]
    hip_yaw_q = asset.data.joint_pos[:, hip_yaw_cfg.joint_ids]

    torso_excess = torch.clamp(torch.abs(torso_q) - max_yaw, min=0.0)
    hip_excess = torch.clamp(torch.abs(hip_yaw_q) - max_yaw, min=0.0)
    return standing_mask * (torch.sum(torch.square(torso_excess), dim=-1) + torch.sum(torch.square(hip_excess), dim=-1))


def standing_still_statue_bonus(
    env,
    min_root_z: float = 0.60,
    max_root_z: float = 0.70,
    min_time_s: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="head_roll_link"),
) -> torch.Tensor:
    """Substantial reward for true motionless standing like a statue after recovery:
    - Root height within [0.60m, 0.70m]
    - Head height within [1.12m, 1.26m]
    - Linear velocity < 0.06 m/s
    - Angular velocity < 0.12 rad/s
    """
    step_time = env.episode_length_buf * env.step_dt
    time_mask = torch.clamp((step_time - min_time_s) / 0.3, min=0.0, max=1.0)

    asset: Articulation = env.scene[asset_cfg.name]
    root_z = asset.data.root_pos_w[:, 2]
    head_z = asset.data.body_pos_w[:, head_cfg.body_ids[0], 2]

    in_height = (root_z >= min_root_z) & (root_z <= max_root_z) & (head_z >= 1.12) & (head_z <= 1.26)

    lin_vel = torch.norm(asset.data.root_lin_vel_w, dim=-1)
    ang_vel = torch.norm(asset.data.root_ang_vel_w, dim=-1)

    stillness = torch.exp(-25.0 * torch.square(lin_vel)) * torch.exp(-10.0 * torch.square(ang_vel))
    return time_mask * in_height.float() * stillness



