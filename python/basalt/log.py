#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import ubjson
import json
import os

import numpy as np

from collections import Mapping
from munch import Munch
from munch import munchify


class ExecutionStats(Munch):

    def __init__(self, path):
        data = self._load(path)
        if data is None:
            Munch.__init__(self)
        else:
            Munch.__init__(self, data)

    def _load(self, path):

        if path.endswith("ubjson"):
            with open(path, 'rb') as f:
                data = ubjson.load(f)
        else:
            with open(path, 'r') as f:
                data = json.load(f)

        if isinstance(data, Mapping):
            data = self._convert(data)

        return munchify(data)

    def _convert(self, data):

        data_new = dict()

        for k, v in data.items():
            if k.endswith("__values"):
                continue  # skip; processed together with __index
            elif k.endswith("__index"):
                idx = v
                values = np.array(data[k.replace("__index", "__values")])
                # convert to list of arrays according to start indices
                res = np.split(values, idx[1:])
                if all(len(res[0]) == len(x) for x in res):
                    res = np.array(res)
                data_new[k.replace("__index", "")] = res
            else:
                data_new[k] = np.array(v)

        return data_new

    def _is_imu(self):
        return len(self.marg_ev[0]) == 15


def detect_log_path(dir, basename):

    for ext in ["ubjson", "json"]:
        path = os.path.join(dir, basename + "." + ext)
        if os.path.isfile(path):
            return path

    return None


def load_execution_stats(dir, basename):

    path = detect_log_path(dir, basename)

    if path is not None:
        return ExecutionStats(path)
    else:
        return None


class Log(Munch):

    @staticmethod
    def load(dir):

        log = Log(all=load_execution_stats(dir, "stats_all"),
                  sums=load_execution_stats(dir, "stats_sums"),
                  vio=load_execution_stats(dir, "stats_vio"))

        if all([v is None for v in log.values()]):
            return None
        else:
            return log

    def __init__(self, *args, **kwargs):
        Munch.__init__(self, *args, **kwargs)

    def duration(self):
        return (self.sums.frame_id[-1] - self.sums.frame_id[0]) * 1e-9
