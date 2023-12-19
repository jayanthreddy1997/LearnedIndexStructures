import os
import yaml


def get_config(config_fp=None):
    config_file = config_fp or ('conf/hpc_config.yml' if 'hpc.nyu.edu' in os.uname().nodename else 'conf/local_config.yml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
