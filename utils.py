
import gymnasium
import torch
from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import highway_env
import gc


from config import CONFIG

training_steps = CONFIG["training_steps"]
num_envs = CONFIG["num_envs"]
training_env = CONFIG["training_env"]
testing_env = CONFIG["testing_env"]
policy = CONFIG["policy"]
env_config = CONFIG["config"]


def make_env(env_id, config):
    def _init():
        env = gymnasium.make(env_id, config=config)
        env = Monitor(env)  
        return env
    return _init

import os
import statistics
from torch.utils.tensorboard import SummaryWriter

def train_and_evaluate(model_class, algo_name, policy, env_config, name_additional_tag=None, tensorboard=True, **model_kwargs):
    print(f"Training {algo_name}...")

    env = SubprocVecEnv([make_env(training_env, env_config) for _ in range(num_envs)])

    log_dir = f"logs/{algo_name}_{training_env}"
    model_save_path = f"models/{algo_name}_{training_env}/model_{policy}"
    results_file = f"models/{algo_name}_{training_env}/results.txt"

    if name_additional_tag:
        log_dir += f"_{name_additional_tag}"
        model_save_path += f"_{name_additional_tag}"

    if tensorboard:
        model = model_class(policy, env, tensorboard_log=log_dir, verbose=1, **model_kwargs)
    else:
        model = model_class(policy, env, verbose=1, **model_kwargs)

    model.learn(training_steps, log_interval=10)
    model.save(model_save_path)
    
    if tensorboard:
        sb3_log_dir = model.logger.dir
        writer = SummaryWriter(sb3_log_dir)

    env.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = model_class.load(model_save_path)
    eval_env = Monitor(gymnasium.make(testing_env, config=env_config))
    rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=10, return_episode_rewards=True)
    
    mean_reward, std_reward = statistics.mean(rewards), statistics.stdev(rewards)
    
    print(f"Testing environment: {testing_env}")
    print("Results:")
    print(f"{algo_name} Mean Reward: {mean_reward} ± {std_reward}")

    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "a") as f:
        f.write(f"{algo_name} - {policy} Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")

    if tensorboard:
        for i, reward in enumerate(rewards):
            writer.add_scalar('Evaluation/Episode Reward', reward, i)
        writer.close()

    eval_env.close()
    del model
    torch.cuda.empty_cache()

    return mean_reward



def evaluate_model(model, env_id, num_episodes):
    env = gymnasium.make(env_id, config=env_config)
    rewards = evaluate_policy(model, env, n_eval_episodes=num_episodes, return_episode_rewards=True)
    mean_reward, std_reward = statistics.mean(rewards), statistics.stdev(rewards)
    print(f"Evaluation results for {env_id}:")
    print(f"Mean Reward: {mean_reward} ± {std_reward}")
    return rewards
