# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.fall_recovery.mdp as mdp
from whole_body_tracking.robots.custom_humanoid import CUSTOM_HUMANOID_CFG


@configclass
class FallRecoverySceneCfg(InteractiveSceneCfg):
    """Configuration for fall recovery flat terrain scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot: ArticulationCfg = CUSTOM_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.13, 0.13, 0.13),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for fall recovery."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.55,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for fall recovery."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for diverse 4-pose fall recovery resets."""

    # Startup material randomization
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 32,
        },
    )

    # Reset diverse lying poses: supine, prone, lateral left, lateral right
    reset_fall_pose = EventTerm(
        func=mdp.reset_diverse_fall_poses,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    """Reward terms for fall recovery."""

    # Primary goal: target bounded range [1.12m, 1.26m] for HEAD height (eliminates jumping bug)
    head_height = RewTerm(
        func=mdp.head_height_range,
        weight=6.0,
        params={
            "min_nominal": 1.12,
            "max_nominal": 1.26,
            "asset_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Secondary goal: target bounded range [0.58m, 0.70m] for PELVIS height
    root_height = RewTerm(
        func=mdp.root_height_range,
        weight=4.0,
        params={"min_nominal": 0.58, "max_nominal": 0.70},
    )

    # Center of mass balance: centers pelvis horizontally over both feet for rock-solid stability
    com_support = RewTerm(
        func=mdp.com_support_alignment,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Torso upright alignment (gated by head height >= 0.45m)
    upright = RewTerm(
        func=mdp.upright_posture_bonus,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="head_roll_link")},
    )

    # Upper body ground contact penalty (penalize pelvis/torso only once elevated or post-grace)
    upper_contact = RewTerm(
        func=mdp.upper_body_ground_contact_penalty,
        weight=-4.0,
        params={
            "min_head_z": 0.70,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "torso_link"]),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Feet contact bonus once head is elevated to standing zone (>= 0.85m)
    feet_contact = RewTerm(
        func=mdp.feet_contact_when_standing,
        weight=4.0,
        params={
            "min_head_height": 0.85,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Settle down to stable stationary stance once standing
    stand_still = RewTerm(
        func=mdp.stand_still_bonus,
        weight=12.0,
        params={
            "min_head_z": 1.05,
            "max_head_z": 1.26,
            "asset_cfg": SceneEntityCfg("robot"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Active stillness enforcement: penalize linear velocity once standing
    still_lin_vel = RewTerm(
        func=mdp.standing_still_lin_vel_penalty,
        weight=-2.0,
        params={
            "min_head_z": 1.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Active stillness enforcement: penalize angular velocity once standing
    still_ang_vel = RewTerm(
        func=mdp.standing_still_ang_vel_penalty,
        weight=-1.0,
        params={
            "min_head_z": 1.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Active stillness enforcement: penalize joint velocity once standing to eliminate jittering
    still_joint_vel = RewTerm(
        func=mdp.standing_still_joint_vel_penalty,
        weight=-0.20,
        params={
            "min_head_z": 1.05,
            "asset_cfg": SceneEntityCfg("robot"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )

    # Active action attenuation: forces action commands a_t -> 0 once upright to let PD stiffness freeze robot
    action_attenuation = RewTerm(
        func=mdp.standing_action_attenuation,
        weight=-4.0,
        params={
            "min_root_z": 0.55,
            "min_time_s": 1.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Ankle roll margin enforcement: keeps foot soles flat on ground, eliminates side-edge rocking
    ankle_roll_margin = RewTerm(
        func=mdp.ankle_roll_margin_penalty,
        weight=-5.0,
        params={
            "min_root_z": 0.55,
            "min_time_s": 1.4,
            "max_roll": 0.08,
            "asset_cfg": SceneEntityCfg("robot"),
            "ankle_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_roll_joint"]),
        },
    )

    # Torso yaw and hip yaw symmetry lock: eliminates twisted waist contortion
    standing_torso_yaw = RewTerm(
        func=mdp.standing_torso_yaw_penalty,
        weight=-6.0,
        params={
            "min_root_z": 0.55,
            "min_time_s": 1.4,
            "max_yaw": 0.06,
            "asset_cfg": SceneEntityCfg("robot"),
            "torso_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"]),
            "hip_yaw_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"]),
        },
    )

    # Statue stillness bonus: rock-solid motionless standing
    statue_bonus = RewTerm(
        func=mdp.standing_still_statue_bonus,
        weight=15.0,
        params={
            "min_root_z": 0.60,
            "max_root_z": 0.70,
            "min_time_s": 1.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
        },
    )


    # Anti-jumping: strictly penalize airborne leaping
    anti_jumping = RewTerm(
        func=mdp.airborne_jumping_penalty,
        weight=-6.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Vertical leap suppression (eliminates explosive uncontrolled launching)
    vertical_vel = RewTerm(func=mdp.excessive_vertical_vel_penalty, weight=-4.0, params={"max_vz": 0.45})

    # Tumble & spin suppression (forces robot to remain facing forward)
    root_ang_vel = RewTerm(func=mdp.root_ang_vel_l2, weight=-0.05)

    # Action smoothness and energy regularization (allow dynamic power during ground pushup)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )

    # 1. Ground recovery: use hands/wrists to push against floor to rise up
    hands_pushup = RewTerm(
        func=mdp.hands_pushup_support_reward,
        weight=3.0,
        params={
            "max_head_z": 0.85,
            "max_time_s": 2.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_wrist_.*"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 2. Anti-lunge / anti-split: strictly penalize feet split apart in X (forces side-by-side standing)
    feet_alignment = RewTerm(
        func=mdp.standing_feet_alignment_penalty,
        weight=-3.0,
        params={
            "min_head_z": 0.90,
            "max_split_x": 0.08,
            "target_width_y": 0.20,
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
            "feet_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # 3. Knee symmetry & extension: strictly penalize one-knee-bent lunge stance
    knee_symmetry = RewTerm(
        func=mdp.standing_knee_symmetry_penalty,
        weight=-3.0,
        params={
            "min_head_z": 0.90,
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "left_knee_cfg": SceneEntityCfg("robot", joint_names=["left_knee_joint"]),
            "right_knee_cfg": SceneEntityCfg("robot", joint_names=["right_knee_joint"]),
        },
    )

    # 4. Attractor to nominal symmetrical upright standing pose (straight legs, arms at side)
    nominal_posture = RewTerm(
        func=mdp.nominal_standing_posture_bonus,
        weight=8.0,
        params={
            "min_head_z": 0.90,
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 5. Knee extension progress: actively rewards extending legs to full upright stance
    knee_extension = RewTerm(
        func=mdp.standing_knee_extension_progress,
        weight=6.0,
        params={
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
            "left_knee_cfg": SceneEntityCfg("robot", joint_names=["left_knee_joint"]),
            "right_knee_cfg": SceneEntityCfg("robot", joint_names=["right_knee_joint"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 6. Knee ground contact penalty: strictly forbid kneeling/shins on floor after initial pushup (t > 1.4s)
    knee_ground_contact = RewTerm(
        func=mdp.knee_ground_contact_penalty,
        weight=-6.0,
        params={
            "min_time_s": 1.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_knee_link", ".*_ankle_pitch_link"]),
        },
    )

    # 7. Knee crouch penalty: strictly penalize knees bent (> 0.45 rad) after 1.4s, forcing upright legs
    knee_crouch = RewTerm(
        func=mdp.standing_knee_crouch_penalty,
        weight=-5.0,
        params={
            "max_knee_angle": 0.45,
            "min_time_s": 1.4,
            "left_knee_cfg": SceneEntityCfg("robot", joint_names=["left_knee_joint"]),
            "right_knee_cfg": SceneEntityCfg("robot", joint_names=["right_knee_joint"]),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 8. Posture deviation penalty: strictly penalize deviation from nominal standing pose after 1.4s
    # Completely prevents waist yaw twisting (e.g. 120 deg) and sideways leg splits
    posture_deviation = RewTerm(
        func=mdp.standing_posture_deviation_penalty,
        weight=-6.0,
        params={
            "min_time_s": 1.4,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 9. DOF joint limits penalty: penalize joints approaching their physical limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Terminations for fall recovery."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fall_after_recovery = DoneTerm(
        func=mdp.fall_after_recovery,
        params={
            "grace_time_s": 3.2,
            "min_head_z": 0.88,
            "min_root_z": 0.42,
            "asset_cfg": SceneEntityCfg("robot"),
            "head_cfg": SceneEntityCfg("robot", body_names="head_roll_link"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "torso_link", "head_roll_link"]),
        },
    )


@configclass
class CustomHumanoidFallRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for fall recovery environment."""

    scene: FallRecoverySceneCfg = FallRecoverySceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 8.0  # 8.0s (1.5s recovery + 6.5s sustained rock-solid stationary standing)
        self.sim.dt = 0.005
        self.sim.render_interval = 4

        # Decoupled action authority: full power on rise joints (torso, hip, knee, arms), bounded ankle roll to prevent hard-stop limit chatter
        self.actions.joint_pos.scale = {
            ".*_hip_pitch_joint": 0.55,
            ".*_hip_roll_joint": 0.55,
            ".*_hip_yaw_joint": 0.35,
            ".*_knee_joint": 0.55,
            ".*_ankle_pitch_joint": 0.45,
            ".*_ankle_roll_joint": 0.12,
            "torso_joint": 0.55,
            "head_.*": 0.20,
            ".*_shoulder_.*": 0.55,
            ".*_elbow_joint": 0.55,
            ".*_wrist_joint": 0.30,
        }






@configclass
class CustomHumanoidFallRecoveryEnvCfg_PLAY(CustomHumanoidFallRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.episode_length_s = 10.0
