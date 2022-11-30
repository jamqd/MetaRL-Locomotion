# Import the RL algorithm (Algorithm) we would like to use.
from ray.rllib.algorithms.maml import MAML

# Configure the algorithm.
config = {
    # Environment (RLlib understands openAI gym registered strings).
    "env": "ray.rllib.examples.env.halfcheetah_rand_direc.HalfCheetahRandDirecEnv",
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    "num_envs_per_worker": 20,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
        "free_log_std": True
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
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

# Create our RLlib Trainer.
algo = MAML(config=config)

# Run it for n training iterations. A training iteration includes
# parallel sample collection by the environment workers as well as
# loss calculation on the collected batch and a model update.
for _ in range(100):
    print(algo.train())

# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
algo.evaluate()