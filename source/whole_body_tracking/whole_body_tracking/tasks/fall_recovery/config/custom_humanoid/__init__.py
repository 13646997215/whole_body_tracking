import gymnasium as gym

from . import agents
from whole_body_tracking.tasks.fall_recovery.fall_recovery_env_cfg import (
    CustomHumanoidFallRecoveryEnvCfg,
    CustomHumanoidFallRecoveryEnvCfg_PLAY,
)

gym.register(
    id="CustomHumanoid-FallRecovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CustomHumanoidFallRecoveryEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidFallRecoveryPPORunnerCfg",
    },
)

gym.register(
    id="CustomHumanoid-FallRecovery-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CustomHumanoidFallRecoveryEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CustomHumanoidFallRecoveryPPORunnerCfg",
    },
)
