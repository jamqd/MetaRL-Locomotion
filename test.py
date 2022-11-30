import ray
from ray import tune
from ray import air
from ray.rllib.algorithms.maml import MAMLConfig
from  ray.rllib.examples.env.ant_rand_goal import AntRandGoalEnv
from datetime import datetime

ray.init()
tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 200, "training_iteration": 50},
        local_dir="./results",
        name="batch_tune"
    ),
    param_space={
        "env": "CartPole-v0",
        "framework": "torch",
        "num_gpus": 1,
        "num_workers": 3,
        "lr": 0.0001,
        'sgd_minibatch_size': tune.grid_search([256, 1024, 2048]),
    },
).fit()