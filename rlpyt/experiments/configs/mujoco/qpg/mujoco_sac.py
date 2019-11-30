
import copy

configs = dict()

config = dict(
    agent=dict(
        model_kwargs=None,
        q_model_kwargs=None,
        v_model_kwargs=None,
    ),
    algo=dict(
        discount=0.99,
        batch_size=256,
        replay_ratio=256,
        target_update_tau=0.01,
        target_update_interval=1,
        learning_rate=3e-4,
        reparameterize=True,
        # policy_output_regularization=0.001,
        reward_scale=10,
        target_entropy="auto",
    ),
    env=dict(id="HalfCheetah-v2"),
    # eval_env=dict(id="Hopper-v3"),  # Train script uses "env".
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        log_interval_steps=1e4,
        seed=None
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        max_decorrelation_steps=0,
        eval_n_envs=4,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    ),
)

configs["sac_1M_serial"] = config

config = copy.deepcopy(configs["sac_1M_serial"])
config["algo"]["bootstrap_timelimit"] = True
configs["sac_serial_bstl"] = config
