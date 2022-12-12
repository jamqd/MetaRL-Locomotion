import ray
from ray import air, tune
from ray.rllib.algorithms.maml import MAMLConfig
from  ray.rllib.examples.env.ant_rand_goal import AntRandGoalEnv
from  ray.rllib.examples.env.halfcheetah_rand_direc import HalfCheetahRandDirecEnv
from datetime import datetime
from ray.tune.stopper import CombinedStopper, TrialPlateauStopper, MaximumIterationStopper

from ray.rllib.examples.env.cartpole_mass import CartPoleMassEnv
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

ray.init(log_to_driver=False)
env = HalfCheetahRandDirecEnv()
tune.register_env("HalfCheetahRandDirecEnv", lambda env_ctx: env)

config = (MAMLConfig()
            .training(
                inner_adaptation_steps=1,
                maml_optimizer_steps=5,
                inner_lr=0.1,
                model={
                    "fcnet_hiddens": [64, 64],
                    "free_log_std" : True
                }
            )
            .framework(framework="torch")
            .resources(num_gpus=1)
            .environment(env="HalfCheetahRandDirecEnv")
            .rollouts(num_rollout_workers=2, horizon=1000, rollout_fragment_length=100)
            .debugging(log_level="ERROR")
            
)

# algo = config.build()


# for i in range(1000):
#     print("Iteration: ", i)
#     info = algo.train()
    
#     metrics = [
#         'adaptation_delta'
#     ]
#     print(" ".join([str(info[m]) for m in metrics]))
#     print()



param_space = config.to_dict()
param_space["inner_lr"] = tune.grid_search([0.01, 0.05, 0.001])
param_space["inner_adaptation_steps"] = tune.grid_search([2, 4])
param_space["horizon"] = tune.grid_search([200, 500, 1000, 2000])

tune.Tuner(  
    "MAML",
    run_config=air.RunConfig(
        stop=CombinedStopper(
            MaximumIterationStopper(max_iter=1000),
            TrialPlateauStopper("adaptation_delta",  std=10.0)
        ),
        # stop={"training_iteration" : 1000},
        # stop=TrialPlateauStopper("adaptation_delta",  std=10.0),
        local_dir="./results",
        name="maml_hyp_search"
    ),
    param_space=param_space
).fit()



# default_MAML_config = {'_disable_action_flattening': False,
#  '_disable_execution_plan_api': False,
#  '_disable_preprocessor_api': False,
#  '_fake_gpus': False,
#  '_tf_policy_handles_more_than_one_loss': False,
#  'action_space': None,
#  'actions_in_input_normalized': False,
#  'always_attach_evaluation_results': False,
#  'batch_mode': 'complete_episodes',
#  'buffer_size': -1,
#  'callbacks': <class 'ray.rllib.algorithms.callbacks.DefaultCallbacks'>,
#  'clip_actions': False,
#  'clip_param': 0.3,
#  'clip_rewards': None,
#  'collect_metrics_timeout': -1,
#  'compress_observations': False,
#  'create_env_on_driver': True,
#  'custom_eval_function': None,
#  'custom_resources_per_worker': {},
#  'disable_env_checking': False,
#  'eager_max_retraces': 20,
#  'eager_tracing': False,
#  'enable_async_evaluation': False,
#  'enable_connectors': False,
#  'enable_tf1_exec_eagerly': False,
#  'entropy_coeff': 0.0,
#  'env': None,
#  'env_config': {},
#  'env_task_fn': None,
#  'evaluation_config': {},
#  'evaluation_duration': 10,
#  'evaluation_duration_unit': 'episodes',
#  'evaluation_interval': None,
#  'evaluation_num_episodes': -1,
#  'evaluation_num_workers': 0,
#  'evaluation_parallel_to_training': False,
#  'evaluation_sample_timeout_s': 180.0,
#  'exploration_config': {'type': 'StochasticSampling'},
#  'explore': True,
#  'export_native_model_files': False,
#  'extra_python_environs_for_driver': {},
#  'extra_python_environs_for_worker': {},
#  'fake_sampler': False,
#  'framework': 'tf',
#  'gamma': 0.99,
#  'grad_clip': None,
#  'horizon': None,
#  'ignore_worker_failures': False,
#  'in_evaluation': False,
#  'inner_adaptation_steps': 1,
#  'inner_lr': 0.1,
#  'input': 'sampler',
#  'input_config': {},
#  'input_evaluation': -1,
#  'keep_per_episode_custom_metrics': False,
#  'kl_coeff': 0.0005,
#  'kl_target': 0.01,
#  'lambda': 1.0,
#  'learning_starts': -1,
#  'local_tf_session_args': {'inter_op_parallelism_threads': 8,
#                            'intra_op_parallelism_threads': 8},
#  'log_level': 'WARN',
#  'log_sys_usage': True,
#  'logger_config': None,
#  'logger_creator': None,
#  'lr': 0.001,
#  'maml_optimizer_steps': 5,
#  'metrics_episode_collection_timeout_s': 60.0,
#  'metrics_num_episodes_for_smoothing': 100,
#  'metrics_smoothing_episodes': -1,
#  'min_iter_time_s': -1,
#  'min_sample_timesteps_per_iteration': 0,
#  'min_sample_timesteps_per_reporting': -1,
#  'min_time_s_per_iteration': None,
#  'min_time_s_per_reporting': -1,
#  'min_train_timesteps_per_iteration': 0,
#  'min_train_timesteps_per_reporting': -1,
#  'model': {'_disable_action_flattening': False,
#            '_disable_preprocessor_api': False,
#            '_time_major': False,
#            '_use_default_native_models': False,
#            'attention_dim': 64,
#            'attention_head_dim': 32,
#            'attention_init_gru_gate_bias': 2.0,
#            'attention_memory_inference': 50,
#            'attention_memory_training': 50,
#            'attention_num_heads': 1,
#            'attention_num_transformer_units': 1,
#            'attention_position_wise_mlp_dim': 32,
#            'attention_use_n_prev_actions': 0,
#            'attention_use_n_prev_rewards': 0,
#            'conv_activation': 'relu',
#            'conv_filters': None,
#            'custom_action_dist': None,
#            'custom_model': None,
#            'custom_model_config': {},
#            'custom_preprocessor': None,
#            'dim': 84,
#            'fcnet_activation': 'tanh',
#            'fcnet_hiddens': [256, 256],
#            'framestack': True,
#            'free_log_std': False,
#            'grayscale': False,
#            'lstm_cell_size': 256,
#            'lstm_use_prev_action': False,
#            'lstm_use_prev_action_reward': -1,
#            'lstm_use_prev_reward': False,
#            'max_seq_len': 20,
#            'no_final_linear': False,
#            'post_fcnet_activation': 'relu',
#            'post_fcnet_hiddens': [],
#            'use_attention': False,
#            'use_lstm': False,
#            'vf_share_layers': False,
#            'zero_mean': True},
#  'monitor': -1,
#  'multiagent': {'count_steps_by': 'env_steps',
#                 'observation_fn': None,
#                 'policies': {},
#                 'policies_to_train': None,
#                 'policy_map_cache': None,
#                 'policy_map_capacity': 100,
#                 'policy_mapping_fn': None},
#  'no_done_at_end': False,
#  'normalize_actions': True,
#  'num_consecutive_worker_failures_tolerance': 100,
#  'num_cpus_for_driver': 1,
#  'num_cpus_per_worker': 1,
#  'num_envs_per_worker': 1,
#  'num_gpus': 0,
#  'num_gpus_per_worker': 0,
#  'num_workers': 2,
#  'observation_filter': 'NoFilter',
#  'observation_space': None,
#  'off_policy_estimation_methods': {},
#  'ope_split_batch_by_episode': True,
#  'optimizer': {},
#  'output': None,
#  'output_compress_columns': ['obs', 'new_obs'],
#  'output_config': {},
#  'output_max_file_size': 67108864,
#  'placement_strategy': 'PACK',
#  'postprocess_inputs': False,
#  'preprocessor_pref': 'deepmind',
#  'prioritized_replay': -1,
#  'prioritized_replay_alpha': -1,
#  'prioritized_replay_beta': -1,
#  'prioritized_replay_eps': -1,
#  'recreate_failed_workers': False,
#  'remote_env_batch_wait_ms': 0,
#  'remote_worker_envs': False,
#  'render_env': False,
#  'replay_batch_size': -1,
#  'replay_mode': -1,
#  'replay_sequence_length': None,
#  'restart_failed_sub_environments': False,
#  'rollout_fragment_length': 200,
#  'sample_async': False,
#  'sample_collector': <class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>,
#  'sampler_perf_stats_ema_coef': None,
#  'seed': None,
#  'shuffle_buffer_size': 0,
#  'simple_optimizer': -1,
#  'soft_horizon': False,
#  'sync_filters_on_rollout_workers_timeout_s': 60.0,
#  'synchronize_filters': True,
#  'tf_session_args': {'allow_soft_placement': True,
#                      'device_count': {'CPU': 1},
#                      'gpu_options': {'allow_growth': True},
#                      'inter_op_parallelism_threads': 2,
#                      'intra_op_parallelism_threads': 2,
#                      'log_device_placement': False},
#  'timesteps_per_iteration': -1,
#  'train_batch_size': 32,
#  'use_gae': True,
#  'use_meta_env': True,
#  'validate_workers_after_construction': True,
#  'vf_clip_param': 10.0,
#  'vf_loss_coeff': 0.5,
#  'vf_share_layers': -1}


"""
Act, obs

Ant
Box(-1.0, 1.0, (8,), float32)
(27,)
env = gym.make("Ant-v3")

Half Cheetah

Box(-1.0, 1.0, (6,), float32)
(17,)
env = gym.make("HalfCheetah-v3")

Hopper

Box(-1.0, 1.0, (3,), float32)
(11,)
env = gym.make("Hopper-v3")

Swimmer
Box(-1.0, 1.0, (2,), float32)
(8,)
env = gym.make("Swimmer-v3")

Walker2D
Box(-1.0, 1.0, (6,), float32)
(17,)
env = gym.make("Walker2d-v3")







"""