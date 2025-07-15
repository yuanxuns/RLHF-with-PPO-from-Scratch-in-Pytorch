from src.train.train_ppo import train
from argparse import ArgumentParser

import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/config/ppo.yaml")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    train(config)
    