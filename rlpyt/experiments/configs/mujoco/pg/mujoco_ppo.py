
import copy

configs = dict()

config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        clip_grad_norm=1e6,
        entropy_loss_coeff=0.0,
        gae_lambda=0.95,
        minibatches=32,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=True,
        linear_lr_schedule=True,
        #bootstrap_timelimit=False,
    ),
    env=dict(id="Hopper-v2"),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        log_interval_steps=2048 * 10,
        seed=None
    ),
    sampler=dict(
        batch_T=2048,
        batch_B=1,
        max_decorrelation_steps=0,
    ),
)

configs["ppo_500k_serial"] = config

config = copy.deepcopy(configs["ppo_500k_serial"])

config = copy.deepcopy(configs["ppo_500k_serial"])
config["sampler"]["batch_B"] = 8
config["sampler"]["batch_T"] = 256
configs["ppo_500k_cpu"] = config

config["algo"]["minibatches"] = 1
config["algo"]["epochs"] = 32
configs["ppo_32ep_1mb"] = config
