# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stair climbing configuration for custom humanoid robot."""

import numpy as np
import torch
import trimesh

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import feet_double_flight
from .rough_env_cfg import CustomHumanoidRoughEnvCfg


def humanoid_base_height_relative(
    env,
    target_height: float = 0.65,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Penalize base height relative to the stance (supporting) foot on the terrain/stairs.

    Prevents hyperextension / pelvic thrust by regulating pelvis height
    relative to the actual supporting foot in contact with the step:
    H_rel = Z_pelvis - Z_stance_foot
    Loss = (H_rel - target_height)^2
    """
    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    contact_weights = contacts.float()
    contact_count = contact_weights.sum(dim=1, keepdim=True)
    has_contact = (contact_count > 0).squeeze(1)

    max_contact_z = torch.where(contacts, feet_z, -100.0).max(dim=1)[0]
    fallback_z = torch.min(feet_z, dim=1)[0]
    stance_z = torch.where(has_contact, max_contact_z, fallback_z)

    base_z = asset.data.root_pos_w[:, 2]
    rel_height = base_z - stance_z
    squat_barrier = torch.square(torch.clamp(0.58 - rel_height, min=0.0)) * 25.0
    return torch.square(rel_height - target_height) + squat_barrier


def stair_ascent_push_up(
    env,
    target_vz: float = 0.18,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward pelvis velocity (extending stance leg to climb) during forward motion."""
    asset = env.scene[asset_cfg.name]
    vz = asset.data.root_lin_vel_w[:, 2]
    vx = asset.data.root_lin_vel_w[:, 0]
    return torch.clamp(vz / target_vz, min=0.0, max=1.2) * torch.clamp(vx / 0.20, min=0.0, max=1.0)


def stair_feet_split_bound(
    env,
    max_z_split: float = 0.20,
    max_x_split: float = 0.42,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Strictly penalize feet separation greater than 20cm in Z or 42cm in X on stairs.
    Forces both feet to alternate climbing step-by-step and eliminates trailing foot lag!
    """
    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    feet_x = asset.data.body_pos_w[:, asset_cfg.body_ids, 0]
    max_z = torch.maximum(feet_z[:, 0], feet_z[:, 1])
    on_stairs = max_z >= 0.08

    z_diff = torch.abs(feet_z[:, 0] - feet_z[:, 1])
    z_excess = torch.clamp(z_diff - max_z_split, min=0.0)

    x_diff = torch.abs(feet_x[:, 0] - feet_x[:, 1])
    x_excess = torch.clamp(x_diff - max_x_split, min=0.0)

    penalty = (z_excess * 6.0 + torch.square(z_excess * 3.0)) + (x_excess * 6.0 + torch.square(x_excess * 3.0))
    return on_stairs.float() * penalty


def stair_double_stance_penalty(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Penalize lingering in double support (both feet in contact) while on the staircase steps.
    Forces immediate weight transfer onto the climbing step and unloading the rear foot!
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    both_touching = (contacts[:, 0] > 0.0) & (contacts[:, 1] > 0.0)

    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]
    max_z = torch.maximum(feet_z[:, 0], feet_z[:, 1])
    min_z = torch.minimum(feet_z[:, 0], feet_z[:, 1])

    # Active during the staircase ascent (from first step 0.08m to top step 0.90m)
    on_stairs = (max_z >= 0.08) & (min_z <= 0.90)

    return both_touching.float() * on_stairs.float()


def humanoid_posture_biomechanics(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
) -> torch.Tensor:
    """Biomechanical posture regularization: allows natural forward trunk lean for ascending stairs."""
    asset = env.scene[asset_cfg.name]
    gx = asset.data.projected_gravity_b[:, 0]
    gy = asset.data.projected_gravity_b[:, 1]

    # Torso uprightness: allow natural forward lean for climbing 21.8 deg stairs (gx in [0.0, 0.35] ~ 0 to 20.5 deg)
    pitch_excess_fwd = torch.clamp(gx - 0.35, min=0.0)
    pitch_excess_bwd = torch.clamp(-gx, min=0.0)
    forward_barrier = pitch_excess_fwd * 8.0 + torch.square(pitch_excess_fwd * 14.0)
    backward_barrier = pitch_excess_bwd * 8.0 + torch.square(pitch_excess_bwd * 14.0)
    pitch_err = forward_barrier + backward_barrier
    roll_err = torch.square(gy * 3.0)

    return pitch_err + roll_err


def forward_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Linear forward progress reward: actively drives forward locomotion along +X at stair climbing speed."""
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_w[:, 0]
    return torch.clamp(vx, min=-0.1, max=0.50)


def standing_still_penalty(
    env,
    min_vel: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize robot for standing still / stalling when commanded to walk forward."""
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_w[:, 0]
    stall = torch.clamp(min_vel - vx, min=0.0)
    return stall


def stair_touchdown_event_bonus(
    env,
    min_dx: float = 0.14,
    min_dz: float = 0.05,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Event-triggered touchdown bonus: rewards planting swing foot onto the next step.

    Triggered ONLY on the exact simulation step of touchdown (current_contact_time <= 1.5 * step_dt)
    when preceding swing air time >= 0.14s, foot is >= 14cm ahead of stance foot,
    and on stairs (elevation >= 0.06m) landed >= 5cm higher than stance foot (step-by-step climb)!
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    last_air = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    touchdown_dt = 1.5 * env.step_dt
    c0_touchdown = (contact_time[:, 0] > 0.0) & (contact_time[:, 0] <= touchdown_dt) & (last_air[:, 0] >= 0.14)
    c1_touchdown = (contact_time[:, 1] > 0.0) & (contact_time[:, 1] <= touchdown_dt) & (last_air[:, 1] >= 0.14)

    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_x = asset.data.body_pos_w[:, asset_cfg.body_ids, 0]

    max_z = torch.maximum(feet_z[:, 0], feet_z[:, 1]) - env.scene.env_origins[:, 2]
    on_stairs = max_z >= 0.06

    # Foot 0 stepping up onto next step ahead of Foot 1
    f0_reach = feet_x[:, 0] - feet_x[:, 1] >= min_dx
    f0_climb = feet_z[:, 0] - feet_z[:, 1] >= min_dz
    f0_valid = f0_reach & (~on_stairs | f0_climb)
    bonus_0 = c0_touchdown.float() * f0_valid.float()

    # Foot 1 stepping up onto next step ahead of Foot 0
    f1_reach = feet_x[:, 1] - feet_x[:, 0] >= min_dx
    f1_climb = feet_z[:, 1] - feet_z[:, 0] >= min_dz
    f1_valid = f1_reach & (~on_stairs | f1_climb)
    bonus_1 = c1_touchdown.float() * f1_valid.float()

    return bonus_0 + bonus_1


def foot_air_time_hard_cap(
    env,
    max_air_time: float = 0.45,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Strictly penalize holding a foot in the air longer than 0.45s.
    Kills the flamingo hovering cheat at its mathematical root while allowing 0.33s stair swing clearance!
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(air_time - max_air_time, min=0.0)
    return torch.sum(torch.square(excess * 10.0), dim=-1)


def foot_stance_time_hard_cap(
    env,
    max_stance_time: float = 0.52,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strictly penalize standing on one foot longer than 0.52s during forward locomotion.
    Prevents single-leg stalling and forces weight transfer / alternating stepping.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    asset = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_w[:, 0]
    is_walking = (vx > 0.10).float().unsqueeze(-1)

    excess = torch.clamp(contact_time - max_stance_time, min=0.0) * is_walking
    return torch.sum(torch.square(excess * 8.0), dim=-1)


def stair_swing_clearance(
    env,
    target_height: float = 0.14,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Reward vertical clearance of the swing foot relative to stance foot to clear 12cm risers."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    f0_swing = (1.0 - contacts[:, 0]) * contacts[:, 1] * (air_time[:, 0] >= 0.04).float()
    f1_swing = (1.0 - contacts[:, 1]) * contacts[:, 0] * (air_time[:, 1] >= 0.04).float()

    dz_0 = feet_z[:, 0] - feet_z[:, 1]
    dz_1 = feet_z[:, 1] - feet_z[:, 0]
    z_score_0 = torch.clamp((dz_0 + 0.02) / target_height, min=0.0, max=1.2)
    z_score_1 = torch.clamp((dz_1 + 0.02) / target_height, min=0.0, max=1.2)

    return f0_swing * z_score_0 + f1_swing * z_score_1


def stair_swing_forward_reach(
    env,
    target_reach: float = 0.30,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Reward the swing foot for actively advancing forward past the stance foot towards the next step."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]

    asset = env.scene[asset_cfg.name]
    feet_x = asset.data.body_pos_w[:, asset_cfg.body_ids, 0]
    feet_vx = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 0]

    f0_swing = (1.0 - contacts[:, 0]) * contacts[:, 1] * (air_time[:, 0] >= 0.04).float()
    f1_swing = (1.0 - contacts[:, 1]) * contacts[:, 0] * (air_time[:, 1] >= 0.04).float()

    reach_0 = torch.clamp((feet_x[:, 0] - feet_x[:, 1] + 0.10) / (target_reach + 0.10), min=0.0, max=1.2)
    reach_1 = torch.clamp((feet_x[:, 1] - feet_x[:, 0] + 0.10) / (target_reach + 0.10), min=0.0, max=1.2)

    vx_0 = torch.clamp(feet_vx[:, 0] / 0.5, min=0.0, max=1.0)
    vx_1 = torch.clamp(feet_vx[:, 1] / 0.5, min=0.0, max=1.0)

    score_0 = reach_0 * (0.5 + 0.5 * vx_0)
    score_1 = reach_1 * (0.5 + 0.5 * vx_1)

    return f0_swing * score_0 + f1_swing * score_1


def pelvis_forward_overextension_penalty(
    env,
    max_lead: float = 0.22,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Penalize pelvis extending more than 22cm ahead of the front foot to prevent toppling over forward."""
    asset = env.scene[asset_cfg.name]
    feet_x = asset.data.body_pos_w[:, asset_cfg.body_ids, 0]
    front_foot_x = torch.maximum(feet_x[:, 0], feet_x[:, 1])
    pelvis_x = asset.data.root_pos_w[:, 0]

    excess = torch.clamp(pelvis_x - front_foot_x - max_lead, min=0.0)
    return excess * 6.0 + torch.square(excess * 12.0)


def trailing_leg_drag_penalty(
    env,
    max_lag: float = 0.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Penalize the trailing foot lagging more than max_lag (28cm) behind the pelvis.
    Mathematically eliminates trailing leg dragging and knee floor-collision!
    """
    asset = env.scene[asset_cfg.name]
    feet_x = asset.data.body_pos_w[:, asset_cfg.body_ids, 0]
    rear_foot_x = torch.minimum(feet_x[:, 0], feet_x[:, 1])
    pelvis_x = asset.data.root_pos_w[:, 0]

    excess = torch.clamp(pelvis_x - rear_foot_x - max_lag, min=0.0)
    return excess * 6.0 + torch.square(excess * 12.0)


def double_foot_same_stair_penalty(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
) -> torch.Tensor:
    """Penalize landing both feet on the exact same stair step while actively climbing stairs.

    Forces true human-like 1-foot-per-step alternating stair climbing!
    Allows normal double-support walking on flat ground and top platform.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()
    both_touching = (contacts[:, 0] > 0.0) & (contacts[:, 1] > 0.0)

    asset = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2:3]

    # Only active when feet are elevated on the stair steps (between 0.06m and 0.90m)
    min_z = torch.minimum(feet_z[:, 0], feet_z[:, 1])
    max_z = torch.maximum(feet_z[:, 0], feet_z[:, 1])
    on_stairs = (min_z >= 0.06) & (max_z <= 0.90)

    # Same elevation (< 0.06m difference) when both feet touch ground on stairs
    z_diff = torch.abs(feet_z[:, 0] - feet_z[:, 1])
    same_step = (z_diff < 0.06).float()

    return both_touching.float() * on_stairs.float() * same_step


def knee_biomechanical_range(
    env,
    min_flexion: float = 0.15,
    max_flexion: float = 1.45,
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
) -> torch.Tensor:
    """Penalize knee hyperextension / peg-legging (< 0.15 rad) and excessive hyperflexion (> 1.45 rad).
    Enforces compliant, symmetric, human-like knee flexion on both legs!
    """
    asset = env.scene[knee_cfg.name]
    knee_pos = asset.data.joint_pos[:, knee_cfg.joint_ids]
    excess_high = torch.clamp(knee_pos - max_flexion, min=0.0)
    excess_low = torch.clamp(min_flexion - knee_pos, min=0.0)
    return torch.sum(
        excess_high * 10.0 + torch.square(excess_high * 20.0) + excess_low * 10.0 + torch.square(excess_low * 20.0),
        dim=-1,
    )


def stance_knee_collapse_penalty(
    env,
    max_stance_knee: float = 0.85,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
) -> torch.Tensor:
    """Penalize knee flexion > 0.85 rad (~49 deg) on the stance leg while in ground contact.
    Allows natural knee flexion for step absorption while preventing deep knee collapse (>50 deg)!
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0).float()

    asset = env.scene[knee_cfg.name]
    knee_pos = asset.data.joint_pos[:, knee_cfg.joint_ids]
    excess = torch.clamp(knee_pos - max_stance_knee, min=0.0)
    return torch.sum(contacts * (excess * 6.0 + torch.square(excess * 12.0)), dim=-1)


def corridor_lateral_stability(
    env,
    max_y_dev: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Strictly lock the robot to the corridor centerline (Y = 0) with zero lateral drift."""
    asset = env.scene[asset_cfg.name]
    vy = asset.data.root_lin_vel_b[:, 1]
    y_dev = torch.abs(asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    excess_y = torch.clamp(y_dev - max_y_dev, min=0.0)
    return torch.square(vy * 2.0) + torch.square(y_dev * 4.0) + excess_y * 10.0 + torch.square(excess_y * 20.0)


##
# Custom Straight Stairs Corridor Terrain Generator
# Layout: Approach Runway (2.0m) -> 8 Ascending Steps (12cm, +0.96m) -> Top Platform (2.8m) -> 8 Descending Steps (12cm, -0.96m) -> Exit Runway (4.0m)
# Includes continuous 35cm side safety curbs along both edges to prevent falling off corridor!
##

def straight_stairs_terrain(
    difficulty: float, cfg: SubTerrainBaseCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a multi-tier straight staircase track corridor with safety side curbs.

    Spawn origin: X = 1.65m, Y = 2.0m (0.35m before the first step for immediate climbing entry).
    """
    step_height = getattr(cfg, "step_height", 0.12)
    step_width = getattr(cfg, "step_width", 0.30)
    num_steps = getattr(cfg, "num_steps", 8)
    corridor_width = cfg.size[1]
    depth_pad = 0.5  # Solid thickness below ground to prevent physics penetration

    approach_len = 2.0
    plat_len = 2.8
    exit_len = 4.0

    curb_w = 0.15
    curb_extra_h = 0.35

    meshes = []

    def add_segment(x_c, x_len, z_top):
        box_h = depth_pad + z_top
        # Main central walkway
        b = trimesh.creation.box(
            (x_len, corridor_width, box_h),
            trimesh.transformations.translation_matrix([x_c, corridor_width / 2.0, z_top - box_h / 2.0]),
        )
        meshes.append(b)
        # Left safety curb
        b_curb_l = trimesh.creation.box(
            (x_len, curb_w, box_h + curb_extra_h),
            trimesh.transformations.translation_matrix([x_c, curb_w / 2.0, (z_top + curb_extra_h) - (box_h + curb_extra_h) / 2.0]),
        )
        meshes.append(b_curb_l)
        # Right safety curb
        b_curb_r = trimesh.creation.box(
            (x_len, curb_w, box_h + curb_extra_h),
            trimesh.transformations.translation_matrix([x_c, corridor_width - curb_w / 2.0, (z_top + curb_extra_h) - (box_h + curb_extra_h) / 2.0]),
        )
        meshes.append(b_curb_r)

    # 1. Approach runway [0.0, 2.0]
    add_segment(approach_len / 2.0, approach_len, 0.0)

    # 2. Ascending steps
    for i in range(num_steps):
        x_c = approach_len + (i + 0.5) * step_width
        z_top = (i + 1) * step_height
        add_segment(x_c, step_width, z_top)

    # 3. Top landing platform
    x_plat_start = approach_len + num_steps * step_width
    plat_z = num_steps * step_height
    add_segment(x_plat_start + plat_len / 2.0, plat_len, plat_z)

    # 4. Descending steps
    x_desc_start = x_plat_start + plat_len
    for i in range(num_steps):
        x_c = x_desc_start + (i + 0.5) * step_width
        z_top = (num_steps - 1 - i) * step_height
        add_segment(x_c, step_width, z_top)

    # 5. Exit runway
    x_exit_start = x_desc_start + num_steps * step_width
    add_segment(x_exit_start + exit_len / 2.0, exit_len, 0.0)

    # Spawn origin: 0.35m before the first step for immediate climbing entry
    origin = np.array([1.65, corridor_width / 2.0, 0.0])
    return meshes, origin


@configclass
class MeshStraightStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a straight staircase corridor mesh terrain."""

    function = straight_stairs_terrain
    step_height: float = 0.12
    step_width: float = 0.30
    num_steps: int = 8


STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=0.0,
    num_rows=6,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "straight_stairs": MeshStraightStairsTerrainCfg(
            proportion=1.0,
            step_height=0.12,
            step_width=0.30,
            num_steps=8,
        ),
    },
)


@configclass
class CustomHumanoidStairsEnvCfg(CustomHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Use specialized straight stairs corridor generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
        self.episode_length_s = 35.0

        # Pure proprioceptive blind locomotion (90-dim observation space)
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # Commands: strictly forward velocity along +X with active heading lock (corridor centerline)
        self.commands.base_velocity.ranges.lin_vel_x = (0.42, 0.48)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.heading_control_stiffness = 2.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        # Reset pose: strictly facing forward along positive X (towards stairs)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        # Reset joints: strictly nominal pose (1.0, 1.0) - eliminates spawn stumble!
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # Decoupled action scale: 0.38 for hip pitch to allow 34cm forward reach; 0.30 for knee; 0.25 nominal
        self.actions.joint_pos.scale = {
            ".*_hip_pitch_joint": 0.38,
            ".*_knee_joint": 0.30,
            ".*_hip_roll_joint": 0.25,
            ".*_hip_yaw_joint": 0.25,
            ".*_ankle_.*": 0.25,
            "torso_joint": 0.25,
            "head_.*": 0.25,
            ".*_shoulder_.*": 0.25,
            ".*_(elbow|wrist)_joint": 0.25,
        }

        # Rewards:
        # 1. Forward velocity tracking & Heading turn rate tracking
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # 1b. Active linear forward momentum & strict stall prevention (breaks standing still local minimum!)
        self.rewards.forward_progress = RewTerm(func=forward_progress, weight=2.5)
        self.rewards.standing_still_penalty = RewTerm(func=standing_still_penalty, weight=-2.5)

        # 2. Base height regulation relative to stance foot on the stairs (nominal 0.65m + anti-squat barrier)
        self.rewards.base_height = RewTerm(
            func=humanoid_base_height_relative,
            weight=-4.0,
            params={
                "target_height": 0.65,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # 2b. Stance leg vertical push-up: rewards extending supporting leg to push pelvis up each 12cm riser
        self.rewards.stair_ascent_push_up = RewTerm(func=stair_ascent_push_up, weight=2.5)

        # 3. Biomechanical Posture: strictly enforces upright posture
        self.rewards.flat_orientation_l2 = RewTerm(
            func=humanoid_posture_biomechanics,
            weight=-1.5,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "knee_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
            },
        )

        # 4. Anti-bunny-hopping: strictly penalize both feet in the air (forces alternating bipedal steps)
        self.rewards.double_flight = RewTerm(
            func=feet_double_flight,
            weight=-4.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )

        # 5. Feet air time: completely remove continuous air time reward to eliminate flamingo hovering exploit!
        self.rewards.feet_air_time = None

        # 5b. Event-triggered stair touchdown bonus: rewards planting swing foot onto next step
        self.rewards.stair_touchdown_bonus = RewTerm(
            func=stair_touchdown_event_bonus,
            weight=15.0,
            params={
                "min_dx": 0.14,
                "min_dz": 0.05,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # 5c. Foot air time hard cap: strictly penalizes holding foot in air > 0.45s
        self.rewards.foot_air_time_hard_cap = RewTerm(
            func=foot_air_time_hard_cap,
            weight=-3.0,
            params={
                "max_air_time": 0.45,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )

        # 5d. Foot stance time hard cap: strictly penalizes single-stance stall > 0.50s during walking
        self.rewards.foot_stance_time_hard_cap = RewTerm(
            func=foot_stance_time_hard_cap,
            weight=-2.0,
            params={
                "max_stance_time": 0.50,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # 6. Stair swing clearance: rewards lifting swinging foot 14cm to clear 12cm risers
        self.rewards.stair_swing_clearance = RewTerm(
            func=stair_swing_clearance,
            weight=3.5,
            params={
                "target_height": 0.14,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # 6b. Stair swing forward reach: actively drives the swing foot 30cm forward past stance foot to next step
        self.rewards.stair_swing_forward_reach = RewTerm(
            func=stair_swing_forward_reach,
            weight=4.0,
            params={
                "target_reach": 0.30,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # 7. Same stair penalty: penalizes having both feet on the exact same stair step (enforces 1-foot-per-step!)
        self.rewards.same_stair_penalty = RewTerm(
            func=double_foot_same_stair_penalty,
            weight=-4.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # 8. Strict alternating stair split bound: prevents feet split > 22cm in Z or > 45cm in X
        self.rewards.stair_split_penalty = RewTerm(
            func=stair_feet_split_bound,
            weight=-3.0,
            params={
                "max_z_split": 0.22,
                "max_x_split": 0.45,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # 8b. Double stance is natural in bipedal walking - do NOT penalize
        self.rewards.stair_double_stance = None

        # 9. Knee Biomechanical Range: prevents hyperextension (< 0.15 rad) and hyperflexion (> 1.45 rad)
        self.rewards.knee_biomechanical_range = RewTerm(
            func=knee_biomechanical_range,
            weight=-3.0,
            params={
                "min_flexion": 0.15,
                "max_flexion": 1.45,
                "knee_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
            },
        )

        # 10. Stance Knee Anti-Collapse: forces stance leg to extend (< 0.85 rad) and lift the body up
        self.rewards.stance_knee_collapse = RewTerm(
            func=stance_knee_collapse_penalty,
            weight=-3.0,
            params={
                "max_stance_knee": 0.85,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "knee_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"]),
            },
        )

        # 11. Corridor Lateral Stability: strictly penalizes lateral drift and sideways velocity
        self.rewards.lateral_stability = RewTerm(
            func=corridor_lateral_stability,
            weight=-3.0,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # 12. Trailing leg anti-drag: strictly penalizes trailing foot lagging > 26cm behind pelvis
        self.rewards.trailing_leg_drag = RewTerm(
            func=trailing_leg_drag_penalty,
            weight=-3.5,
            params={
                "max_lag": 0.26,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # 12b. Pelvis forward overextension: strictly prevent pelvis lunging > 22cm ahead of front foot
        self.rewards.pelvis_overextension = RewTerm(
            func=pelvis_forward_overextension_penalty,
            weight=-3.0,
            params={
                "max_lead": 0.22,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            },
        )

        # Terminations: terminate on base or torso contact with stairs/ground
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_link", "torso_link"]

        # Dynamics and stability
        self.rewards.dof_pos_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_ankle_.*", ".*_knee_joint", ".*_hip_.*"]
        )
        self.rewards.dof_pos_limits.weight = -3.0
        self.rewards.lin_vel_z_l2 = None  # Crucial: Allow natural vertical ascent/descent on stairs!
        self.rewards.feet_slide.weight = -0.25   # Prevent slipping on stair step edges
        self.rewards.termination_penalty.weight = -100.0


@configclass
class CustomHumanoidStairsEnvCfg_PLAY(CustomHumanoidStairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_heading_envs = 1.0
        self.commands.base_velocity.heading_control_stiffness = 2.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.debug_vis = False
