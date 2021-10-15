#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import math
import numpy as np


def best_two_non_repeating(array, reverse=False):
    if reverse:
        best = -math.inf
        second = -math.inf
        for v in array:
            if v > best:
                second = best
                best = v
            elif v < best and v > second:
                second = v
    else:
        best = math.inf
        second = math.inf
        for v in array:
            if v < best:
                second = best
                best = v
            elif v > best and v < second:
                second = v

    return best, second


def format_ratio(val, val_ref=None, decimals=0):
    if val_ref == 0:
        return "{}".format(math.inf)
    else:
        if val_ref is not None:
            val = float(val) / float(val_ref)
        return "{:.{prec}f}".format(val, prec=decimals)


def format_ratio_percent(val, val_ref=None, decimals=0):
    if val_ref == 0:
        return "{}".format(val)
    else:
        if val_ref is not None:
            val = float(val) / float(val_ref)
        val = 100 * val
        return "{:.{prec}f}%".format(val, prec=decimals)


def rotation2d(theta_deg):
    theta = np.radians(theta_deg)

    R = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))

    return R
