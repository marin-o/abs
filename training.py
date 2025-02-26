from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from config import CONFIG
from utils import train_and_evaluate

training_steps = CONFIG["training_steps"]
num_envs = CONFIG["num_envs"]
training_env = CONFIG["training_env"]
testing_env = CONFIG["testing_env"]
policy = CONFIG["policy"]
env_config = CONFIG["config"]
name_additional_tag = CONFIG["name_additional_tag"]

# === TRAINING LOOPS ===
if __name__ == "__main__":
    # --- Train A2C ---
    train_and_evaluate(A2C, "A2C", policy, env_config,
                       learning_rate=0.00015078369731868298, gamma=0.9663796739439382, n_steps=64, name_additional_tag='merge_optimized')

    # # --- Train PPO ---
    # train_and_evaluate(PPO, "PPO", policy, env_config,
    #                    learning_rate=3e-4, gamma=0.99, n_steps=128, batch_size=64, name_additional_tag=name_additional_tag)

    # # --- Train QRDQN ---
    # train_and_evaluate(QRDQN, "QRDQN", policy, env_config,
    #                    learning_rate=1e-4, gamma=0.99, batch_size=64, buffer_size=50_000, name_additional_tag=name_additional_tag)

    print("All models trained and saved successfully!")
