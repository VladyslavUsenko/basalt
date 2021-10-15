#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import os
import json
import platform
import subprocess
import re
import numpy as np

from munch import munchify


def copy_subdict(d, keys):
    res = dict()
    for k in keys:
        if k in d:
            res[k] = d[k]
    return res


def load_json_if_exists(filepath):
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return munchify(json.load(f))
    return None


def load_text_if_exists(filepath):
    if os.path.isfile(filepath):
        with open(filepath, 'r') as f:
            return f.read()
    return None


def load_trajectory_tum(filepath):
    # first row is header
    # format for each row is: ts x y z qz qy qz qw
    traj = np.loadtxt(filepath, delimiter=" ", skiprows=1)
    # return just translation for now
    traj = traj[:, 1:4]
    return traj


def load_trajectory_tum_if_exists(filepath):
    if os.path.isfile(filepath):
        return load_trajectory_tum(filepath)
    return None


def os_open_file(filepath):
    if platform.system() == 'Darwin':
        subprocess.call(('open', filepath))
    elif platform.system() == 'Windows':
        os.startfile(filepath)
    else:
        subprocess.call(('xdg-open', filepath))


# key for 'human' sorting
def alphanum(key):

    def convert(text):
        return float(text) if text.isdigit() else text

    return [convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
