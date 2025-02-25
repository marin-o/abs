import ray
from ray import train
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray import tune
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
from stable_baselines3.common.monitor import Monitor
import gymnasium



from config import CONFIG
from utils import train_and_evaluate, make_env

training_steps = CONFIG["training_steps"]
num_envs = CONFIG["num_envs"]
training_env = CONFIG["training_env"]
testing_env = CONFIG["testing_env"]
policy = CONFIG["policy"]
env_config = CONFIG["config"]

print(env_config)

def train_a2c(config):
    """Training function for A2C with Ray Tune."""
    env = gymnasium.make(training_env, config=config)
    env = Monitor(env)  
    print(env)
    model = A2C(
        policy=policy,
        env=env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        n_steps=config["n_steps"],
        vf_coef=config["vf_coef"],
        tensorboard_log="./a2c_tune_logs/",
    )
    
    model.learn(total_timesteps=training_steps)
    
    mean_reward, _ = train_and_evaluate(model, testing_env, num_episodes=10)
    
    tune.report(mean_reward=mean_reward) 

def main():
    ray.init(ignore_reinit_error=True, num_gpus=1)
    
    # Define hyperparameter search space
    search_space = {
        "learning_rate": tune.uniform(1e-5, 1e-1),
        "gamma": tune.choice([0.95, 0.99]),
        "ent_coef": tune.uniform(0.0, 0.01),
        "n_steps": tune.choice([5, 10, 20]),
        "vf_coef": tune.uniform(0.1, 1.0),
    }
    
    scheduler = ASHAScheduler(
        metric="mean_reward", mode="max", max_t=training_steps, grace_period=10000, reduction_factor=2
    )
    
    analysis = tune.run(
        train_a2c,
        config=search_space,
        num_samples=10,  
        scheduler=scheduler,
        verbose=1,
        max_concurrent_trials=1,
    )
    
    print("Best config: ", analysis.get_best_config(metric="mean_reward", mode="max"))

    with open("best_config_a2c.txt", "w") as f:
        f.write(str(analysis.get_best_config(metric="mean_reward", mode="max")))

if __name__ == "__main__":
    main()