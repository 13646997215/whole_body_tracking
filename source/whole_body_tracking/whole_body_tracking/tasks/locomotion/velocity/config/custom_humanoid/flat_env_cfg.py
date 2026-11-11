# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .rough_env_cfg import CustomHumanoidRoughEnvCfg


import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor


def feet_double_flight(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize both feet in the air simultaneously (jumping / bunny-hopping).

    In bipedal walking, at least one foot must maintain ground contact at all times.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    both_in_air = (torch.sum(contacts.int(), dim=1) == 0).float()
    return both_in_air


@configclass
class CustomHumanoidFlatEnvCfg(CustomHumanoidRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Change terrain to flat plane
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove height scanner for flat terrain
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # Anti-bunny-hopping: strictly penalize both feet in the air (forces alternating bipedal steps)
        self.rewards.double_flight = RewTerm(
            func=feet_double_flight,
            weight=-2.5,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )

        # Base height regulation: keep torso at nominal walking height (~0.66m), strictly penalizing squatting/jumping
        self.rewards.base_height = RewTerm(
            func=mdp.base_height_l2,
            weight=-2.0,
            params={"target_height": 0.66, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Suppress vertical bouncing velocity (hopping has huge vz, walking vz ~ 0)
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.1

        # Moderate roll tilt penalty (allows natural ~2 deg pelvic sway during single-leg stance)
        self.rewards.flat_orientation_l2.weight = -1.0

        # Rewards for alternating single-stance foot clearance
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_slide.weight = -0.25
        self.rewards.joint_deviation_hip.weight = -0.2

        # Velocity and heading tracking
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        )

        # Omnidirectional commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Push disturbance for robust balance (anticipates external pushes)
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(8.0, 12.0),
            params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
        )


@configclass
class CustomHumanoidFlatEnvCfg_PLAY(CustomHumanoidFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Single environment for visual playback
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 60.0

        # Disable observation corruption and push perturbation for clean evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Fixed forward gait command for visual evaluation
        self.commands.base_velocity.ranges.lin_vel_x = (0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False
