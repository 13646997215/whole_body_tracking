from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from whole_body_tracking.robots.custom_humanoid import CUSTOM_HUMANOID_CFG
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class CustomHumanoidTrackingFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CUSTOM_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Decoupled expressive action scaling: gives upper body full amplitude (up to 150 deg)
        # to match real Mocap dance gestures while maintaining stable bipedal foot support
        self.actions.joint_pos.scale = {
            ".*_shoulder_pitch_joint": 1.30,
            ".*_shoulder_roll_joint": 1.20,
            ".*_shoulder_yaw_joint": 1.00,
            ".*_elbow_joint": 1.30,
            ".*_wrist_joint": 0.80,
            "torso_joint": 0.60,
            "head_.*": 0.30,
            ".*_hip_pitch_joint": 0.70,
            ".*_knee_joint": 0.75,
            ".*_hip_roll_joint": 0.35,
            ".*_hip_yaw_joint": 0.35,
            ".*_ankle_pitch_joint": 0.45,
            ".*_ankle_roll_joint": 0.35,
        }

        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.motion_file = "motions/custom_humanoid_dance1.npz"
        self.commands.motion.body_names = [
            "base_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_link",
        ]

        # Rewards: High-fidelity joint tracking + balanced action rate for fluid coordination
        self.rewards.motion_joint_pos = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=4.0,
            params={"command_name": "motion", "std": 0.25},
        )
        self.rewards.motion_body_pos = RewTerm(
            func=mdp.motion_relative_body_position_error_exp,
            weight=3.0,
            params={"command_name": "motion", "std": 0.20},
        )
        self.rewards.motion_body_ori.weight = 2.5
        self.rewards.motion_global_anchor_pos.weight = 1.5
        self.rewards.action_rate_l2.weight = -0.020  # Damps high-frequency jerk for smooth, coordinated limb motion
        self.rewards.joint_limit.weight = -2.0

        # Relax early terminations to allow full exploration of dynamic dance moves
        self.terminations.anchor_pos.params["threshold"] = 0.50
        self.terminations.anchor_ori.params["threshold"] = 1.20
        self.terminations.ee_body_pos.params["threshold"] = 0.50
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_link",
            "right_wrist_link",
        ]
        self.episode_length_s = 20.0


@configclass
class CustomHumanoidTrackingFlatEnvCfg_PLAY(CustomHumanoidTrackingFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 220.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.motion.debug_vis = False
        if hasattr(self.scene, "contact_forces") and self.scene.contact_forces is not None:
            self.scene.contact_forces.debug_vis = False
