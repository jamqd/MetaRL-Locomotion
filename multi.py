import ray
from ray import air, tune
from ray.rllib.algorithms.maml import MAML
from ray.rllib.algorithms.maml import MAMLConfig
from  ray.rllib.examples.env.ant_rand_goal import AntRandGoalEnv
from  ray.rllib.examples.env.halfcheetah_rand_direc import HalfCheetahRandDirecEnv
from datetime import datetime
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper, MaximumIterationStopper

from ray.rllib.examples.env.cartpole_mass import CartPoleMassEnv

# from ray.rllib.examples.env.pendulum_mass import PendulumMassEnv
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 


from gym.envs.classic_control.pendulum import PendulumEnv
from gym.utils import EzPickle
import numpy as np

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv

from MujocoMultipleEnv import MujocoMultipleEnv

ray.init(log_to_driver=False)

env = MujocoMultipleEnv()
tune.register_env("MujocoMultipleEnv", lambda env_ctx: env)

HalfCheetahRandDirecEnvConfig = {
                                  "env": "MujocoMultipleEnv",
                                  "num_workers": 2,
                                  "num_envs_per_worker": 20,
                                  "framework": "torch",
                                  "model": {
                                      "fcnet_hiddens": [128, 128],
                                      "free_log_std": True
                                  },
                                  "evaluation_num_workers": 1,
                                  "evaluation_config": {
                                      "render_env": False,
                                  },
                                  "rollout_fragment_length": 100,
                                  "inner_adaptation_steps": 4,
                                  "maml_optimizer_steps": 5,
                                  "gamma": 0.99,
                                  "lambda": 1.0,
                                  "lr": 0.001,
                                  "vf_loss_coeff": 0.5,
                                  "clip_param": 0.3,
                                  "kl_target": 0.01,
                                  "kl_coeff": 0.0005,
                                  "num_gpus": 1,
                                  "inner_lr": 0.1,
                                  "clip_actions": False,
                                  "horizon": 500
                              }

print("Runnign multi")
algo = MAML(config=HalfCheetahRandDirecEnvConfig)

for i in range(2000):
    print("Iteration: ", i)
    info = algo.train()
    
    metrics = [
        'adaptation_delta'
    ]
    print(" ".join([str(info[m]) for m in metrics]))
    print()



# config = (MAMLConfig()
#             .training(
#                 inner_adaptation_steps=1,
#                 maml_optimizer_steps=5,
#                 inner_lr=0.1,
#                 model={
#                     "fcnet_hiddens": [64, 64],
#                     "free_log_std" : True
#                 }
#             )
#             .framework(framework="torch")
#             .resources(num_gpus=1)
#             .environment(env=AntRandGoalEnv)
#             .rollouts(num_rollout_workers=2, horizon=1000, rollout_fragment_length=100)
#             .debugging(log_level="ERROR")
            
# )

# config = (MAMLConfig()
#             .training(
#                 inner_adaptation_steps=1,
#                 maml_optimizer_steps=5,
#                 inner_lr=0.1,
#                 model={
#                     "fcnet_hiddens": [64, 64],
#                     "free_log_std" : True
#                 }
#             )
#             .framework(framework="torch")
#             .resources(num_gpus=1)
#             .rollouts(num_rollout_workers=2, horizon=1000, rollout_fragment_length=100)
#             .debugging(log_level="ERROR")
            
# )

# algo = config.build()


# for i in range(5):
#     print("Iteration: ", i)
#     info = algo.train()
    
#     metrics = [
#         'adaptation_delta'
#     ]
#     print(" ".join([str(info[m]) for m in metrics]))
#     print()