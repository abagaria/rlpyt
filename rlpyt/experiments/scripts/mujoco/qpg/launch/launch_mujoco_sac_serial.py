from random import randint
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_sac_serial.py"
affinity_code = encode_affinity(
    n_cpu_core=12,
    n_gpu=4,
    contexts_per_gpu=3,
    # hyperthread_offset=2,
    # n_socket=1,
    cpu_per_run=1,
    set_affinity=True
)
runs_per_setting = 1
default_config_key = "sac_1M_serial"
experiment_title = "sac_mujoco"
variant_levels = list()

env_ids = ["Walker2d-v2", "Hopper-v2", "Humanoid-v2"]  # , "Swimmer-v3"]
values = list(zip(env_ids))
dir_names = ["env_{}_{}".format(*v, randint(0, 1000)) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

n_experiments = 5
# Within a variant level, list each combination explicitly.
seeds = [randint(100, 100000) for _ in range(n_experiments)]
values = list(zip(seeds))
dir_names = [experiment_title + "_{}seed".format(*v) for v in values]
keys = [("runner", "seed")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
