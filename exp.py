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


class PendulumMassEnv(PendulumEnv, TaskSettableEnv):
    """PendulumMassEnv varies the weight of the pendulum

    Tasks are defined to be weight uniformly sampled between [0.5,2]
    """

    def sample_tasks(self, n_tasks):
        # Sample new pendulum masses (random floats between 0.5 and 2).
        return np.random.uniform(low=0.5, high=2.0, size=(n_tasks,))

    def set_task(self, task):
        """
        Args:
            task: Task of the meta-learning environment (here: mass of
                the pendulum).
        """
        # self.m is the mass property of the pendulum.
        self.m = task

    def get_task(self):
        """
        Returns:
            float: The current mass of the pendulum (self.m in the PendulumEnv
                object).
        """
        return self.m

# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('/home/jamqd/git/MetaRL-Locomotion/exp/')

ray.init(log_to_driver=False)

env = PendulumMassEnv()
tune.register_env("PendulumMassEnv", lambda env_ctx: env)

# AntRandGoalEnvConfig = {
#                         "env": "ray.rllib.examples.env.ant_rand_goal.AntRandGoalEnv",
#                         "num_workers": 2,
#                         "framework": "torch",
#                         "rollout_fragment_length": 200,
#                         "num_envs_per_worker": 20,
#                         "inner_adaptation_steps": 2,
#                         "maml_optimizer_steps": 5,
#                         "gamma": 0.99,
#                         "lambda": 1.0,
#                         "lr": 0.001,
#                         "vf_loss_coeff": 0.5,
#                         "clip_param": 0.3,
#                         "kl_target": 0.01,
#                         "kl_coeff": 0.0005,
#                         "num_gpus": 1,
#                         "inner_lr": 0.03,
#                         "clip_actions": False,
#                         "model" : {
#                             "fcnet_hiddens": [64, 64],
#                             "free_log_std": True
#                         },
#                         "horizon": 100
#                       } 



# print("Running ant")
# algo = MAML(config=AntRandGoalEnvConfig)
# for i in range(500):
#     print("Iteration: ", i)
#     info = algo.train()
    
#     metrics = [
#         'adaptation_delta'
#     ]
#     print(" ".join([str(info[m]) for m in metrics]))
#     print()

# ray.rllib.examples.env.pendulum_mass.PendulumMassEnv
PendulumMassEnvConfig = {
                        "env": "PendulumMassEnv",
                        "num_workers": 2,
                        "framework": "torch", 
                        "rollout_fragment_length": 200,
                        "num_envs_per_worker": 10,
                        "inner_adaptation_steps": 1,
                        "maml_optimizer_steps": 5,
                        "gamma": 0.99,
                        "lambda": 1.0,
                        "lr": 0.001,
                        "vf_loss_coeff": 0.5,
                        "clip_param": 0.3,
                        "kl_target": 0.01,
                        "kl_coeff": 0.001,
                        "num_gpus": 1,
                        "inner_lr": 0.03,
                        "explore": True,
                        "clip_actions": False,
                        "model": {
                            "fcnet_hiddens": [64, 64],
                            "free_log_std": True
                          },
                         "horizon": 100 
                        }


# print("Running pend")
# algo = MAML(config=PendulumMassEnvConfig)
# for i in range(500):
#     print("Iteration: ", i)
#     info = algo.train()
    
#     metrics = [
#         'adaptation_delta'
#     ]
#     print(" ".join([str(info[m]) for m in metrics]))
#     print()


# configure environments here.
HalfCheetahRandDirecEnvConfig = {
                                  "env": "ray.rllib.examples.env.halfcheetah_rand_direc.HalfCheetahRandDirecEnv",
                                  "num_workers": 2,
                                  "num_envs_per_worker": 20,
                                  "framework": "torch",
                                  "model": {
                                      "fcnet_hiddens": [64, 64],
                                      "free_log_std": True
                                  },
                                  "evaluation_num_workers": 1,
                                  "evaluation_config": {
                                      "render_env": False,
                                  },
                                  "rollout_fragment_length": 100,
                                  "inner_adaptation_steps": 1,
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
                                  "horizon": 100
                              }

print("Runnign cheetah")
algo = MAML(config=HalfCheetahRandDirecEnvConfig)

for i in range(1000):
    print("Iteration: ", i)
    info = algo.train()
    
    metrics = [
        'adaptation_delta'
    ]
    print(" ".join([str(info[m]) for m in metrics]))
    print()