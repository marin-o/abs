import gc
import torch
import json
import os

from stable_baselines3 import PPO, A2C
from sb3_contrib import QRDQN
from config import CONFIG
CONFIG["training_steps"] = 50000
from utils import train_and_evaluate
from skopt import gp_minimize
from skopt.space import Real, Categorical


LOG_FILE = "best_params_log.json"



def optimize_hyperparameters(model_class, model_name, policy, env_config, name_additional_tag):
    def objective(params):
        if model_class == A2C:
            learning_rate, gamma, use_rms_prop = params
        else:
            learning_rate, gamma = params[:2]

        # Set arguments dynamically based on model type
        train_kwargs = {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "name_additional_tag": name_additional_tag,
            "tensorboard": False
        }

        if model_class == PPO:
            train_kwargs.update({"batch_size": 64, "n_steps": 128})
        elif model_class == QRDQN:
            train_kwargs.update({"batch_size": 64, "buffer_size": 50000})
        elif model_class == A2C:
            train_kwargs["use_rms_prop"] = use_rms_prop

        evaluation_result = train_and_evaluate(model_class, model_name, policy, env_config, **train_kwargs)

        return -evaluation_result

    # Define search space
    search_space = [
        Real(1e-5, 1e-2, prior='log-uniform', name='learning_rate'),
        Real(0.8, 0.999, name='gamma')
    ]
    
    if model_class == A2C:
        search_space.append(Categorical([True, False], name='use_rms_prop'))

    res = gp_minimize(objective, search_space, n_calls=CONFIG['optim']['n_calls'], random_state=0)

    return res


def log_best_params(model_name, res):
    """Log the final best parameters for a model."""
    best_params = {
        "model_name": model_name,
        "learning_rate": res.x[0],
        "gamma": res.x[1]
    }
    
    if model_name == "A2C":
        best_params["use_rms_prop"] = bool(res.x[2])

    with open(f"best_params_{model_name}.json", "w") as f:
        json.dump(best_params, f)


if __name__ == "__main__":
    for model_class, model_name in [(PPO, "PPO")]: #  ,(A2C, "A2C") , (QRDQN, "QRDQN")
        res = optimize_hyperparameters(model_class, model_name, CONFIG["policy"], CONFIG["config"], CONFIG["name_additional_tag"])
        print(f"Best hyperparameters for {model_name}: Learning Rate: {res.x[0]}, Gamma: {res.x[1]}")

        if model_class == A2C:
            print(f"Use RMSProp: {res.x[2]}")

        log_best_params(model_name, res)

        # Cleanup
        del res
        gc.collect()
        torch.cuda.empty_cache()
