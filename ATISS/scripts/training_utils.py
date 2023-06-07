# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json

import string
import os
import random
import subprocess


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)
