import yaml


def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("Configuration file not found.")
        exit(1)
    except yaml.YAMLError as exc:
        print("Error parsing configuration:", exc)
        exit(1)
