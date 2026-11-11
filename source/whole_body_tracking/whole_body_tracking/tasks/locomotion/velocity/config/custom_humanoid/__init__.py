# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg, stairs_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="CustomHumanoid-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CustomHumanoidFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CustomHumanoidFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidFlatPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.CustomHumanoidRoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidRoughPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-Velocity-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.CustomHumanoidRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidRoughPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-Velocity-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stairs_env_cfg.CustomHumanoidStairsEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidStairsPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-Velocity-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": stairs_env_cfg.CustomHumanoidStairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidStairsPPORunnerCfg",
    },
)
