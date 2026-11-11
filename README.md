# BeyondMimic Motion Tracking Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

### Alternative Implementations

- There is an alternative reproduction of BeyondMimic in [mjlab](https://github.com/mujocolab/mjlab), a new Isaac Lab-style manager API powered by MuJoCo-Warp for RL and robotics research. See the implementation [here](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)


- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".


- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name} --headless
```

This will automatically upload the processed motion file to the WandB registry with output name {motion_name}.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
python scripts/replay_npz.py --registry_name={your-organization}-org/wandb-registry-motions/{motion_name}
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

- Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along
with a unique 8-character identifier. Note that run_name is different from run_path.

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.

## Custom Humanoid Extension (This Fork)

This fork adds a custom 26-DoF humanoid and three task families on top of the original BeyondMimic tracking code:

| Task | Gym ID | Description |
| --- | --- | --- |
| Dance tracking | `CustomHumanoid-Tracking-Flat-v0` | Motion imitation from a local `.npz` motion file |
| Velocity locomotion | `CustomHumanoid-Velocity-Flat-v0` / `-Rough-v0` / `-Stairs-v0` | Flat / rough terrain / 12 cm stairs walking |
| Fall recovery | `CustomHumanoid-FallRecovery-v0` | Stand up from prone/supine and remain upright |

### Assets

Robot assets are intentionally not committed to keep the repository lightweight.
Place the custom humanoid asset package at:

```
source/whole_body_tracking/whole_body_tracking/assets/custom_robot/
├── robot_urdf.usd
├── robot_urdf.urdf
├── config.yaml
├── configuration/   # robot_urdf_base.usd, physics and sensor layers
└── meshes/          # STL/OBJ link meshes
```

The Unitree description meshes used by the original G1 tasks live in
`source/whole_body_tracking/whole_body_tracking/assets/unitree_description/`.

### Usage

Training (requires Isaac Lab v2.1.0 activated in the current Python environment):

```bash
# Dance tracking from a local motion file
python scripts/rsl_rl/train.py --task CustomHumanoid-Tracking-Flat-v0 \
    --motion_file motions/custom_humanoid_dance1.npz --num_envs 4096 --headless

# Locomotion / stairs / fall recovery
python scripts/train_locomotion.py --task CustomHumanoid-Velocity-Stairs-v0 --num_envs 1024 --headless
python scripts/train_locomotion.py --task CustomHumanoid-FallRecovery-v0 --num_envs 1024 --headless
```

Playback:

```bash
python scripts/rsl_rl/play.py --task CustomHumanoid-Tracking-Flat-Play-v0 \
    --checkpoint exported_models/custom_humanoid/custom_humanoid_tracking_final.pt
python scripts/play_locomotion.py --task CustomHumanoid-Velocity-Stairs-Play-v0 \
    --checkpoint exported_models/custom_humanoid/custom_humanoid_stairs_final.pt
```

### Utility Scripts

| Script | Purpose |
| --- | --- |
| `scripts/train_locomotion.py` | Train velocity / stairs / fall-recovery policies |
| `scripts/play_locomotion.py` | Play locomotion checkpoints with configurable camera |
| `scripts/test_custom_robot.py` | Spawn the custom humanoid and validate the asset |
| `scripts/convert_custom_robot_npz.py` | Convert a dance CSV into the robot's `.npz` format |
| `scripts/replay_npz.py` | Replay a converted motion in Isaac Sim |
| `scripts/diagnose_tracking.py` | Evaluate tracking RMSE and survival |
| `scripts/diagnose_fall_recovery.py` | Evaluate fall-recovery telemetry |
| `scripts/diagnose_telemetry.py` | Generic telemetry evaluation |
| `scripts/debug_fall_breakdown.py` | Break down fall/recovery failure modes |
| `scripts/run_continuous_pipeline.py` | Run stairs → fall recovery → tracking sequentially |
| `scripts/continuous_overhaul_suite.py` | Continuous training/export/verification suite |
| `scripts/autonomous_iterate_and_verify.py` | Autonomous train → diagnose iteration loop |

`run_continuous_pipeline.py`, `continuous_overhaul_suite.py`, and
`autonomous_iterate_and_verify.py` use the current Python interpreter and the
repository root by default. Override them with the `PYTHON_BIN` and
`WBT_WORKSPACE` environment variables.

### Notes

- `logs/`, `wandb/`, `outputs/`, `exported_models/`, `urdf/`, `motions/`, and `*.mp4` are ignored by git.
- The original BeyondMimic documentation above applies to the G1 tracking pipeline.
