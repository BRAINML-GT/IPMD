from utils.common import EvalCallback, linear_schedule, plot_costs
import os
import datetime

import gym
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.sac.sac import SAC 

from apmd_off.iapmd import IPMD as IPMD_OFF
from apmd_off.apmd import PMD as APMD_OFF

plt.style.use('ggplot')

def run_off_policy_iapmd(env_id):
    expert_samples_replay_buffer_loc = f"expert_models/HalfCheetah-v4-2022-11-28 23:19:30/halfcheetah-v4.pkl"

    env = make_vec_env(env_id, n_envs=1)
    ipmd_model = IPMD_OFF("MlpPolicy",
                          env,
                          gamma=1.0, #discount factor
                          verbose=1,
                          batch_size=512,
                          train_freq=1,
                          gradient_steps=5,
                          expert_replay_buffer_loc=expert_samples_replay_buffer_loc)

    eval_env = make_vec_env(env_id, n_envs=1)
    logtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'logs/iapmd/10traj/{env_id}-{logtime}/',
                                 log_path=f'logs/iapmd/10traj/{env_id}-{logtime}/', eval_freq=1000)

    ipmd_model.learn(total_timesteps=2e6, log_interval=10, callback=eval_callback)

def run_off_policy_apmd(env_id):
    env = make_vec_env(env_id, n_envs=1)
    ipmd_model = APMD_OFF("MlpPolicy", 
                            env, 
                            gamma=1.0, #discount factor
                            verbose=1, 
                            learning_rate=linear_schedule(5e-4),
                            batch_size=512, 
                            train_freq=10,
                            ent_coef=0.1,
                            gradient_steps=1,
                            buffer_size=5000000)

    eval_env = make_vec_env(env_id, n_envs=1)
    logtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'logs/apmd/sac-style/{env_id}-{logtime}/',
                                 log_path=f'logs/apmd/sac-style/{env_id}-{logtime}/', eval_freq=1e5)
    ipmd_model.learn(total_timesteps=3e6, log_interval=5, callback=eval_callback)

if __name__ == "__main__":
    run_off_policy_iapmd('HalfCheetah-v4')
        

