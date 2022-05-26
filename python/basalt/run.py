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
from collections import Mapping

from .log import Log

from .util import load_json_if_exists
from .util import load_text_if_exists
from .util import load_trajectory_tum_if_exists


class Run:
    """Loads files from a single run of an experiment from a folder (config, status, output, log, ...)

    A single run is one invocation of odometry with a specific config on a specific sequence.
    This is meant to be used on directories created with the 'generate-batch-configs' and 'run-all-in' scripts.
    It's best-effort, loading as many of the files as are present.
    """

    def __init__(self, dirpath, seq_name_mapping):
        self.dirpath = dirpath

        self.config = load_json_if_exists(os.path.join(dirpath, 'basalt_config.json'))
        self.output = load_text_if_exists(os.path.join(dirpath, 'output.log'))
        self.status = load_text_if_exists(os.path.join(dirpath, 'status.log'))
        self.traj_est = load_trajectory_tum_if_exists(os.path.join(dirpath, 'trajectory.txt'))
        self.traj_gt = load_trajectory_tum_if_exists(os.path.join(dirpath, 'groundtruth.txt'))

        self.log = Log.load(dirpath)

        self.seq_name = self._infer_sequence_name(self.config, dirpath, seq_name_mapping)

        print("loaded {} from '{}'".format(self.seq_name, dirpath))

    def is_imu(self):
        return self.config.batch_run.args["use-imu"] in [1, "1", True, "true"]

    def is_failed(self):
        if self.log is None:
            return True
        else:
            return "Completed" not in self.status

    def failure_str(self):
        if not self.is_failed():
            return ""
        if self.output:
            if "Some of your processes may have been killed by the cgroup out-of-memory handler" in self.output:
                return "OOM"
            if "DUE TO TIME LIMIT" in self.output:
                return "OOT"
        return "x"

    @staticmethod
    def _infer_sequence_name(config, dirpath, name_mapping):
        """Tries to infer the sequence name from the config, or falls back to the parent folder name"""
        seq_name = ""
        try:
            type = config.batch_run.args["dataset-type"]
            path = config.batch_run.args["dataset-path"]
            seq_name = os.path.basename(path)
            if type == "euroc":
                if seq_name.startswith("dataset-"):
                    # assume tumvi
                    seq_name = seq_name.replace("dataset-", "tumvi-").split("_")[0]
                else:
                    # assume euroc
                    s = seq_name.split("_")
                    seq_name = "euroc{}{}".format(s[0], s[1])
            elif type == "kitti":
                # assume kitti
                seq_name = "kitti{}".format(seq_name)
        except:
            pass

        # Fallback to detecting the sequence name base on the last component of the parent folder. This is intended
        # to work for run folders created with the 'generate-batch-configs' script, assuming the sequence is the
        # last component in '_batch.combinations'.
        if seq_name == "":
            seq_name = os.path.basename(dirpath).split("_")[-1]

        # optionally remap the sequence name to something else as defined in the experiments config
        if isinstance(name_mapping, Mapping) and seq_name in name_mapping:
            seq_name = name_mapping[seq_name]

        return seq_name

    @staticmethod
    def is_run_dir(dirpath):
        """Returns True if the folder may be a run directory, based on the present files

        This is intended to be used for auto-detecting run directories in a file tree.
        """

        files = [
            'status.log',
            'slurm-output.log',
            'output.log',
            'stats_all.ubjson',
            'stats_all.json',
            'stats_sums.ubjson',
            'stats_sums.json',
            'stats_vio.ubjson',
            'stats_vio.json',
        ]
        for f in files:
            if os.path.isfile(os.path.join(dirpath, f)):
                return True
        return False
