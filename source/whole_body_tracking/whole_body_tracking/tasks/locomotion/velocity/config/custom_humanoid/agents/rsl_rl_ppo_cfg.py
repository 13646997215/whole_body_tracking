from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CustomHumanoidRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "custom_humanoid_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class CustomHumanoidFlatPPORunnerCfg(CustomHumanoidRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1000
        self.save_interval = 50
        self.experiment_name = "custom_humanoid_flat"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]


@configclass
class CustomHumanoidStairsPPORunnerCfg(CustomHumanoidRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1200
        self.save_interval = 50
        self.experiment_name = "custom_humanoid_stairs"
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
