from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from config import CONFIG
CONFIG["training_steps"]=50000
from utils import train_and_evaluate
from skopt import gp_minimize
from skopt.space import Real
import json

def optimize_hyperparameters(model_class, model_name, policy, env_config, name_additional_tag):
    def objective(params):
        learning_rate, gamma = params
        evaluation_result = train_and_evaluate(model_class, model_name, policy, env_config,
                                               learning_rate=learning_rate, gamma=gamma, 
                                               name_additional_tag=name_additional_tag, tensorboard=False)
        return -evaluation_result

    search_space = [Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
                    Real(0.8, 0.999, name='gamma')]

    res = gp_minimize(objective, search_space, n_calls=50, random_state=0)
    return res

def log_best_params(model_name, res):
    best_params = {
        "model_name": model_name,
        "learning_rate": res.x[0],
        "gamma": res.x[1]
    }
    with open(f"best_params_{model_name}.json", "w") as f:
        json.dump(best_params, f)

if __name__ == "__main__":
    # Optimize A2C
    res_a2c = optimize_hyperparameters(A2C, "A2C", CONFIG["policy"], CONFIG["config"], CONFIG["name_additional_tag"])
    print(f"Best hyperparameters for A2C: Learning Rate: {res_a2c.x[0]}, Gamma: {res_a2c.x[1]}")
    log_best_params("A2C", res_a2c)

    # # Optimize PPO
    # res_ppo = optimize_hyperparameters(PPO, "PPO", CONFIG["policy"], CONFIG["config"], CONFIG["name_additional_tag"])
    # print(f"Best hyperparameters for PPO: Learning Rate: {res_ppo.x[0]}, Gamma: {res_ppo.x[1]}")
    # log_best_params("PPO", res_ppo)

    # # Optimize QRDQN
    # res_qrdqn = optimize_hyperparameters(QRDQN, "QRDQN", CONFIG["policy"], CONFIG["config"], CONFIG["name_additional_tag"])
    # print(f"Best hyperparameters for QRDQN: Learning Rate: {res_qrdqn.x[0]}, Gamma: {res_qrdqn.x[1]}")
    # log_best_params("QRDQN", res_qrdqn)