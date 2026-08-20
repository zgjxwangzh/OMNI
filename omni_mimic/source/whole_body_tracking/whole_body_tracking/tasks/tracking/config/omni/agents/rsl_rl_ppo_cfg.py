from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class OmniFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    
    # resume from previous checkpoint
    # resume = True

    num_steps_per_env = 24
    max_iterations = 10000000
    save_interval = 2000
    experiment_name = "omni_flat"
    empirical_normalization = True
    runner_class_name = "MotionOnPolicyRunner" # OnPolicyRunner | MotionOnPolicyRunner
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 回退到稳定值
        noise_std_type="scalar",
        actor_hidden_dims=[3072, 1536, 768, 512],  # 回退到稳定网络
        critic_hidden_dims=[3072, 1536, 768, 512],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        # 2026-08-20 最终: lr 对齐跳高训练(1e-3), 加速跳出局部最优
        # 跳高 lr=1e-3 成功收敛 75cm, 跑步用同样值
        learning_rate=1.0e-3,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

LOW_FREQ_SCALE = 0.5


@configclass
class OmniFlatLowFreqPPORunnerCfg(OmniFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)
