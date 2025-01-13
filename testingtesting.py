import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback

# Step 1: Define the custom reward wrapper
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0  # Track Mario's previous x position
        self.prev_lives = 3  # Track Mario's previous lives

    def reset(self, **kwargs):
        self.prev_x = 0
        self.prev_lives = 3
        return self.env.reset(**kwargs)

    def step(self, action):
        # Unpack the five values returned by step()
        obs, reward, done, truncated, info = self.env.step(action)

        # Custom reward logic
        x_pos = info.get("x", 0)  # Mario's x position
        lives = info.get("lives", 3)  # Mario's remaining lives

        # Reward for moving right
        reward += (x_pos - self.prev_x) * 0.1

        # Punish for moving left
        if x_pos < self.prev_x:
            reward -= 1

        # Severely punish for losing a life
        if lives < self.prev_lives:
            reward -= 100

        # Update previous state
        self.prev_x = x_pos
        self.prev_lives = lives

        return obs, reward, done, truncated, info



# Step 2: Load and wrap the retro environment
env_name = "SuperMarioBros-Nes"
env = retro.make(game=env_name, render_mode="rgb_array")
env = CustomRewardWrapper(env)  # Apply the custom reward wrapper
env = DummyVecEnv([lambda: env])  # SB3-compatible wrapper
env = VecTransposeImage(env)  # Convert image format for PyTorch
env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for motion awareness

# Step 3: Define and train the PPO model
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=2.5e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    verbose=1,
    device="auto",
)

# Add a checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./checkpoints", name_prefix="ppo_mario_custom")

# Train the PPO model
model.learn(total_timesteps=25000, callback=checkpoint_callback)
model.save("ppo_mario_custom")
env.close()
print("Model trained and saved successfully!")

# Step 4: Load and test the trained model
del model  # Clear the model
model = PPO.load("ppo_mario_custom")
print("Model loaded successfully!")

# Create a human-rendering environment for testing
env = retro.make(game=env_name, render_mode="human")
env = CustomRewardWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

# Test the trained model
obs = env.reset()
for _ in range(5000):  # Test for 500 steps
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
