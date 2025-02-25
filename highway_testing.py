import gymnasium
import highway_env
from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import time

# Number of training steps for each model
training_steps = 1_000_000

# Number of parallel environments
num_envs = 8

env_name = "merge-v0"


# Config for grayscale observation
config = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),  # Resolution
        "stack_size": 4,                 # Stacked frames
        "weights": [0.2989, 0.5870, 0.1140],  # RGB to grayscale weights
        "scaling": 1.75,                 # Scaling factor
    },
    "policy_frequency": 2
}

discrete_config = {
    "action": {
        "type": "DiscreteAction"
    },
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),  # Resolution
        "stack_size": 4,                 # Stacked frames
        "weights": [0.2989, 0.5870, 0.1140],  # RGB to grayscale weights
        "scaling": 1.75,                 # Scaling factor
    },
    "policy_frequency": 2
}

# Helper function to create parallel environments
def make_env(env_id, config):
    def _init():
        return gymnasium.make(env_id, config=config)
    return _init

def test_model(model_path, model_class, env_config, test_duration=20):
        print(f"Testing model from {model_path}...")
        env = gymnasium.make(env_name, config=env_config, render_mode="human")
        model = model_class.load(model_path)
        start_time = time.time()
        while time.time() - start_time < test_duration:
            done = truncated = False
            obs, info = env.reset()
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                env.render()
            env.close()


if __name__ == "__main__":
    # --- Test Saved Model ---
    
    # Test PPO model
    # test_model(f"highway_ppo_{env_name}/model", PPO, config)

    # Test A2C model
    test_model(f"models/PPO_merge-v0/model_CnnPolicy.zip", A2C, config)

    # Test DQN model
    # test_model(f"highway_dqn_{env_name}/model", QRDQN, discrete_config)
