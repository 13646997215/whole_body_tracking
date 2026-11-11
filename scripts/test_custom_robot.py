"""Script to test loading and simulating the custom humanoid robot in Isaac Lab."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test custom humanoid robot spawning in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from whole_body_tracking.robots.custom_humanoid import CUSTOM_HUMANOID_CFG


@configclass
class CustomRobotTestSceneCfg(InteractiveSceneCfg):
    """Configuration for custom robot test scene."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    robot = CUSTOM_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    from isaaclab.sensors import ContactSensorCfg
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


def main():
    """Main simulation loop for testing 1024 parallel environments."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.005)
    sim = SimulationContext(sim_cfg)

    scene_cfg = CustomRobotTestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    robot: Articulation = scene["robot"]

    print("================================================================================")
    print(f"[SUCCESS] Custom Robot successfully instantiated in Isaac Lab!")
    print(f"  - Number of parallel environments : {scene.num_envs}")
    print(f"  - Number of actuated DOFs         : {robot.num_joints}")
    print(f"  - Number of rigid bodies / links  : {robot.num_bodies}")
    print(f"  - Joint names                     : {robot.joint_names}")
    print("================================================================================")

    # Step simulation for 50 steps
    print(f"[INFO] Stepping physics simulation for 50 steps across {args_cli.num_envs} environments...")
    for step in range(50):
        # Apply zero torque/passive PD holding to verify stable contact and no blow-up
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    contact_sensor = scene["contact_forces"]
    forces = contact_sensor.data.net_forces_w[0]
    for idx, name in enumerate(contact_sensor.body_names):
        f_norm = forces[idx].norm().item()
        if f_norm > 1.0:
            print(f"  [CONTACT] Body '{name}' contact force: {f_norm:.2f} N")

    root_pos = robot.data.root_pos_w[:, :3]
    avg_z = root_pos[:, 2].mean().item()
    print(f"[SUCCESS] Physics simulation completed cleanly!")
    print(f"  - Average base height (Z) across all {args_cli.num_envs} envs: {avg_z:.3f} m")
    print(f"  - VRAM allocated: {torch.cuda.memory_allocated() / (1024**2):.2f} MB")
    print(f"  - VRAM reserved : {torch.cuda.memory_reserved() / (1024**2):.2f} MB")
    print("================================================================================")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush()
    import os
    os._exit(0)
