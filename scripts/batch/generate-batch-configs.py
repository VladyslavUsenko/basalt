#!/usr/bin/env python3
#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#

#
# Generate basalt configurations from a batch config file.
#
# Example:
#     ./generate-batch-config.py /path/to/folder
#
# It looks for the file named `basalt_batch_config.toml` inside the given folder.

import os
import toml
import json
import argparse
from pprint import pprint
from copy import deepcopy
from collections import OrderedDict
import itertools
import shutil
import datetime
import sys


def isdict(o):
    return isinstance(o, dict) or isinstance(o, OrderedDict)


def merge_config(a, b):
    "merge b into a"
    for k, v in b.items():
        if k in a:
            if isdict(v) and isdict(a[k]):
                #print("dict {}".format(k))
                merge_config(a[k], b[k])
            elif not isdict(v) and not isdict(a[k]):
                a[k] = deepcopy(v)
                #print("not dict {}".format(k))
            else:
                raise RuntimeError("Incompatible types for key {}".format(k))
        else:
            a[k] = deepcopy(v)


def save_config(template, configs, combination, path_prefix):
    filename = os.path.join(path_prefix, "basalt_config_{}.json".format("_".join(combination)))
    config = deepcopy(template)
    #import ipdb; ipdb.set_trace()
    for override in combination:
        merge_config(config, configs[override])
    #import ipdb; ipdb.set_trace()
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(filename)


def generate_configs(root_path, cmdline=[], overwrite_existing=False, revision_override=None):

    # load and parse batch config file
    batch_config_path = os.path.join(root_path, "basalt_batch_config.toml")
    template = toml.load(batch_config_path, OrderedDict)
    cfg = template["_batch"]
    del template["_batch"]

    # parse batch configuration
    revision = str(cfg.get("revision", 0)) if revision_override is None else revision_override
    configs = cfg["config"]
    alternatives = cfg.get("alternatives", dict())
    combinations = cfg["combinations"]

    # prepare output directory
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = root_path if revision is None else os.path.join(root_path, revision)
    if overwrite_existing and os.path.exists(outdir):
        print("WARNING: output directory exists, overwriting existing files: {}".format(outdir))
    else:
        os.makedirs(outdir)
    shutil.copy(batch_config_path, outdir)
    with open(os.path.join(outdir, "timestamp"), 'w') as f:
        f.write(date_str)
    with open(os.path.join(outdir, "commandline"), 'w') as f:
        f.write(cmdline)

    # expand single entry in combination array
    def expand_one(x):
        if x in alternatives:
            return alternatives[x]
        elif isinstance(x, list):
            # allow "inline" alternative
            return x
        else:
            return [x]

    def flatten(l):
        for el in l:
            if isinstance(el, list):
                yield from flatten(el)
            else:
                yield el

    # generate all configurations
    for name, description in combinations.items():
        if True or len(combinations) > 1:
            path_prefix = os.path.join(outdir, name)
            if not (overwrite_existing and os.path.exists(path_prefix)):
                os.mkdir(path_prefix)
        else:
            path_prefix = outdir
        expanded = [expand_one(x) for x in description]
        for comb in itertools.product(*expanded):
            # flatten list to allow each alternative to reference multiple configs
            comb = list(flatten(comb))
            save_config(template, configs, comb, path_prefix)


def main():
    cmdline = str(sys.argv)
    parser = argparse.ArgumentParser("Generate basalt configurations from a batch config file.")
    parser.add_argument("path", help="path to look for config and templates")
    parser.add_argument("--revision", help="override revision")
    parser.add_argument("--force", "-f", action="store_true", help="overwrite existing files")
    args = parser.parse_args()
    generate_configs(args.path, cmdline, args.force, args.revision)


if __name__ == "__main__":
    main()
