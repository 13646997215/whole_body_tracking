import gymnasium as gym

from . import agents, flat_env_cfg

gym.register(
    id="CustomHumanoid-Tracking-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CustomHumanoidTrackingFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidTrackingPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-Tracking-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CustomHumanoidTrackingFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-CustomHumanoid-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.CustomHumanoidTrackingFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidTrackingPPORunnerCfg",
    },
)
