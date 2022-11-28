import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
import numpy as np

class MujocoMultipleEnv(MujocoEnv,TaskSettableEnv):
    """ Environment encapsulating Ant, Hopper, Hummanoid, HalfCheetah
    """

    def __init__(self, task_env_names=["HalfCheetah", "Ant", "Hopper"]):
        self.task_env_names = task_env_names
        self.task_envs = []
        for task_env_name in task_env_names:
            self.task_envs.append(gym.make(task_env_name))
        
        self.current_task = self.task_envs[0]

                
    def sample_tasks(self, n_tasks):
        return np.random.choice(range(0,len(self.task_env_names)), (n_tasks,))

    def set_task(self, task_no):
        self.current_task = self.task_envs[task_no]
        
    def get_task(self):
        return self.current_task
    
    def get_action_space(self):
        return self.current_task.action_space
    
    def step(self, action):
        ''' need to truncate action space here '''
        return self.current_task.step(action)
    
    def reset_model(self):
        self.current_task.reset()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

