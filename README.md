# Overview
This repository contains code for training and evaluating reinforcement learning models using the Stable Baselines3 library. The primary focus is on training models for highway and merge environments using various algorithms such as PPO, A2C, and QRDQN.

## Directory Structure
- **logs**: Contains TensorBoard event files for monitoring training progress.
- **models**: Directory where trained models are saved.
- **config.yaml**: Configuration file for training and evaluation settings.
- **config.py**: Script to load configuration from config.yaml.
- **utils.py**: Utility functions for environment creation, training, and evaluation.
- **training.py**: Script to train models using predefined hyperparameters.
- **hyperparameter_optimization.py**: Script for hyperparameter optimization using Bayesian optimization.
- **export_averages_tensorboard.py**: Script to export average rewards from TensorBoard logs to a CSV file.
- **requirements.txt**: List of Python dependencies required for the project.
- **best_params_*.json**: JSON files containing the best hyperparameters found for each model and environment combination.
- **highway.py**: Script to train models specifically for the highway environment.
- **highway_testing.py**: Script to test trained models in the highway environment.

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/marin-o/highway-env-RL-comparison.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure settings: Modify `config.yaml` to set the desired training and evaluation parameters.

## Training Models
To train models using predefined hyperparameters, run:
```
python training.py
```

## Hyperparameter Optimization
To perform hyperparameter optimization, run:
```
python hyperparameter_optimization.py
```
This will optimize hyperparameters for the specified models and save the best parameters in JSON files.

## Exporting Average Rewards
To export average rewards from TensorBoard logs to a CSV file, run:
```
python export_averages_tensorboard.py
```
This will generate `average_values_sorted.csv` containing the average rewards for each configuration.

## Testing Models
To test a trained model in the highway environment, modify and run:
```
python highway_testing.py
```

## Configuration
The `config.yaml` file contains various settings for training and evaluation:
- **training_steps**: Number of training steps.
- **num_envs**: Number of parallel environments.
- **training_env**: Environment used for training.
- **testing_env**: Environment used for evaluation.
- **policy**: Policy type used by the models.
- **name_additional_tag**: Additional tag for naming logs and models.
- **config**: Environment-specific configuration.
- **optim**: Hyperparameter optimization settings.


## Acknowledgements
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Highway-env](https://github.com/eleurent/highway-env)

