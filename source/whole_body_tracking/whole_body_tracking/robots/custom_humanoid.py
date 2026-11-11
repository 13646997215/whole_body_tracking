import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

##
# Actuator dynamic parameters for Custom Humanoid
# Tuned based on robot joint effort limits (Hip: 122Nm, Knee: 122Nm, Ankle: 75Nm, Torso: 94Nm, Arms: 36Nm)
##

# Leg primary drive (Hip / Knee)
STIFFNESS_LEGS = 120.0
DAMPING_LEGS = 6.0
ARMATURE_LEGS = 0.015

# Ankle drive (Pitch / Roll)
STIFFNESS_FEET = 80.0
DAMPING_FEET = 4.0
ARMATURE_FEET = 0.008

# Torso / Waist yaw
STIFFNESS_TORSO = 100.0
DAMPING_TORSO = 5.0
ARMATURE_TORSO = 0.012

# Arms & Wrist
STIFFNESS_ARMS = 40.0
DAMPING_ARMS = 2.0
ARMATURE_ARMS = 0.005

# Head
STIFFNESS_HEAD = 20.0
DAMPING_HEAD = 1.0
ARMATURE_HEAD = 0.002

CUSTOM_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/custom_robot/robot_urdf.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.66),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Legs: slightly bent knees and counter-balanced ankles for stable initial stance
            ".*_hip_pitch_joint": -0.20,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.40,
            ".*_ankle_pitch_joint": -0.20,
            ".*_ankle_roll_joint": 0.0,
            # Torso
            "torso_joint": 0.0,
            # Head
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.0,
            "head_roll_joint": 0.0,
            # Arms: natural relaxed hanging pose
            "left_shoulder_pitch_joint": 0.15,
            "left_shoulder_roll_joint": 0.15,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.35,
            "left_wrist_joint": 0.0,
            "right_shoulder_pitch_joint": 0.15,
            "right_shoulder_roll_joint": -0.15,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.35,
            "right_wrist_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 122.0,
                ".*_hip_roll_joint": 122.0,
                ".*_hip_yaw_joint": 94.0,
                ".*_knee_joint": 122.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 20.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_yaw_joint": 20.0,
                ".*_knee_joint": 25.0,
            },
            stiffness=STIFFNESS_LEGS,
            damping=DAMPING_LEGS,
            armature=ARMATURE_LEGS,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 75.0,
                ".*_ankle_roll_joint": 75.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 20.0,
                ".*_ankle_roll_joint": 15.0,
            },
            stiffness=STIFFNESS_FEET,
            damping=DAMPING_FEET,
            armature=ARMATURE_FEET,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso_joint"],
            effort_limit_sim=94.0,
            velocity_limit_sim=15.0,
            stiffness=STIFFNESS_TORSO,
            damping=DAMPING_TORSO,
            armature=ARMATURE_TORSO,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=[
                "head_yaw_joint",
                "head_pitch_joint",
                "head_roll_joint",
            ],
            effort_limit_sim={
                "head_yaw_joint": 9.0,
                "head_pitch_joint": 12.0,
                "head_roll_joint": 12.0,
            },
            velocity_limit_sim=15.0,
            stiffness=STIFFNESS_HEAD,
            damping=DAMPING_HEAD,
            armature=ARMATURE_HEAD,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_joint",
            ],
            effort_limit_sim=36.0,
            velocity_limit_sim=25.0,
            stiffness=STIFFNESS_ARMS,
            damping=DAMPING_ARMS,
            armature=ARMATURE_ARMS,
        ),
    },
)

# Compute default action scaling per actuator
CUSTOM_ACTION_SCALE = {}
for a in CUSTOM_HUMANOID_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            CUSTOM_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
