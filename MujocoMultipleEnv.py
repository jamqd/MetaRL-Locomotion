import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
import numpy as np
from gym.spaces import Box

class MujocoMultipleEnv(TaskSettableEnv):
    """ Environment encapsulating Ant, Hopper, Hummanoid, HalfCheetah
    """

    def __init__(self):
        self.task_env_names = [
            "Ant-v3", 
            "HalfCheetah-v3", 
            "Hopper-v3",
            "Swimmer-v3",
            "Walker2d-v3"
        ]
        # self.task_envs = []

        self.max_action_dim = 0
        self.max_obs_dim = 0
        for task_env_name in  self.task_env_names:
            env = gym.make(task_env_name)

            self.max_action_dim = max(self.max_action_dim, 
                env.action_space.shape[0])
            self.max_obs_dim = max(self.max_obs_dim, 
                env.observation_space.shape[0])

            # self.task_envs.append(env)
        
        self.current_task_name = self.task_env_names[0]
        # self.current_task = self.task_envs[0]
        # self.sim = self.current_task.sim

        self.action_space = Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.max_action_dim,), 
            dtype = np.float32)

        self.observation_space = Box(
            low=float("-inf"), 
            high=float("inf"), 
            shape=(self.max_obs_dim,), 
            dtype = np.float64)

    def reset(self):
        self.current_task = gym.make(self.current_task_name)
        # self.current_task = self.task_envs[0]
        self.sim = self.current_task.sim
        obs = self.current_task.reset()
        return np.pad(obs, (0, self.max_obs_dim - obs.shape[0]), "constant") 




                
    def sample_tasks(self, n_tasks):
        return np.random.choice(range(0,len(self.task_env_names)), (n_tasks,))

    def set_task(self, task_no):
        self.current_task_name = self.task_env_names[0]
        # self.current_task = self.task_envs[task_no]
        # self.sim = self.current_task.sim
        
    def get_task(self):
        return self.current_task
    
    # def get_action_space(self):
    #     return self.current_task.action_space
    
    def step(self, action):
        ''' need to truncate action space here '''
        action = action[:self.current_task.action_space.shape[0]]
        obs, rew, done, info = self.current_task.step(action)
        obs = np.pad(obs, (0, self.max_obs_dim - obs.shape[0]), "constant") 
        return obs, rew, done, info 
    
    # def reset_model(self):
    #     self.current_task.reset()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

