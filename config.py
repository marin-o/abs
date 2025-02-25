import yaml
import highway_env

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)

CONFIG = load_config()


