import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config')
args = parser.parse_args()
config_path = args.config
config = json.load(open(config_path))
