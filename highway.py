import gymnasium
import highway_env
from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import torch

training_steps = 1_000_000

num_envs = 8

env_name = "parking-v0"

config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),  
        "stack_size": 4,                 
        "weights": [0.2989, 0.5870, 0.1140],  
        "scaling": 1.75,                 
    },
    "policy_frequency": 2
}

def make_env(env_id, config):
    def _init():
        env = gymnasium.make(env_id, config=config)
        env = Monitor(env)  
        return env
    return _init
    

if __name__ == "__main__":
    # --- PPO Training ---
    # print("Training PPO...")
    # env = SubprocVecEnv([make_env(env_name, config) for _ in range(num_envs)])

    # model_ppo = PPO('CnnPolicy', env,
    #                 learning_rate=3e-4,
    #                 batch_size=128,  
    #                 n_steps=1024,    
    #                 gamma=0.99,
    #                 tensorboard_log=f"highway_ppo_{env_name}/",
    #                 verbose=2,
    #                 )
    # model_ppo.learn(int(training_steps))
    # model_ppo.save("highway_ppo/model")
    # del model_ppo
    # torch.cuda.empty_cache()
    # env.close()

    # --- A2C Training ---
    print("Training A2C...")
    env = SubprocVecEnv([make_env(env_name, config) for _ in range(num_envs)])

    model_a2c = A2C('CnnPolicy', env,
                    learning_rate=7e-4,
                    gamma=0.95,
                    n_steps=64,  
                    tensorboard_log=f"highway_a2c_{env_name}/",
                    verbose=2,
                    )
    model_a2c.learn(int(training_steps))
    model_a2c.save("highway_a2c/model")
    del model_a2c
    torch.cuda.empty_cache()
    env.close()

    # --- DQN Training ---
    print("Training DQN...")
    env = SubprocVecEnv([make_env(env_name, config) for _ in range(num_envs)])

    model_dqn = QRDQN('CnnPolicy', env,
                  learning_rate=5e-4,
                  buffer_size=50_000,
                  batch_size=128,
                  gamma=0.8,
                  train_freq=(4, 'step'),
                  gradient_steps=4,
                  tensorboard_log=f"highway_dqn/_{env_name}",
                  verbose=2)
    model_dqn.learn(int(training_steps))
    model_dqn.save("highway_dqn/model")
    del model_dqn
    torch.cuda.empty_cache()
    env.close()

    print("All models trained and saved successfully!")
