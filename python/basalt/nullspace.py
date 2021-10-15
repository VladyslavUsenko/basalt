#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
from .log import ExecutionStats

import argparse

import numpy as np

import matplotlib.pyplot as plt

prop_cycle = plt.rcParams['axes.prop_cycle']

#default_cycler = (cycler(linestyle=['-', '--', ':', '-.']) *
#                  cycler(color=prop_cycle.by_key()['color']))

default_colors = prop_cycle.by_key()['color']


def plot(args):

    log = ExecutionStats(args.path)

    fig, axs = plt.subplots(nrows=6, ncols=1, figsize=(10, 12.0))
    i = 0

    if log._is_imu():
        ns = log.marg_ns[1:, [0, 1, 2, 5]]
    else:
        ns = log.marg_ns[1:, 0:6]
    not_ns = log.marg_ns[1:, 6]

    if True:
        ax = axs[i]
        i += 1

        ax.semilogy(log.marg_ns[1:, 0], ":", label="x", color=default_colors[0])
        ax.semilogy(log.marg_ns[1:, 1], ":", label="y", color=default_colors[0])
        ax.semilogy(log.marg_ns[1:, 2], ":", label="z", color=default_colors[0])

        ax.semilogy(log.marg_ns[1:, 3], ":", label="roll", color=default_colors[1])
        ax.semilogy(log.marg_ns[1:, 4], ":", label="pitch", color=default_colors[1])

        ax.semilogy(log.marg_ns[1:, 5], ":", label="yaw", color=default_colors[2])

        ax.semilogy(log.marg_ns[1:, 6], ":", label="rand", color=default_colors[3])

        #ax.semilogy(np.min(ns, axis=1), "-.", color=default_colors[0])
        #ax.semilogy(np.max(ns, axis=1), ":", color=default_colors[0])
        #ax.semilogy(not_ns, "-", label="foo", color=default_colors[0])
        ax.set_title("nullspace")

        ax.legend(loc="center right")

    ev_all = np.array([x[0:7] for x in log.marg_ev[3:]])
    ev = np.array([x[x > 1e-5][0:7] for x in log.marg_ev[3:]])
    #ev = np.array([x[0:7] for x in log.marg_ev[3:]])

    ev_ns_min = ev[:, 0]
    if log._is_imu():
        print("is vio")
        ev_ns_max = ev[:, 3]
        ev_not_ns = ev[:, 4]
    else:
        print("is vo")
        ev_ns_max = ev[:, 5]
        ev_not_ns = ev[:, 6]

    if True:
        ax = axs[i]
        i += 1

        ax.semilogy(ev_ns_min, "-.", color=default_colors[0])
        ax.semilogy(ev_ns_max, ":", color=default_colors[0])
        ax.semilogy(ev_not_ns, "-", label="foo", color=default_colors[0])

        ax.set_title("eigenvalues (filtered all ev < 1e-5)")
        #ax.set_title("eigenvalues")
        #ax.legend()

    if True:
        ax = axs[i]
        i += 1

        ax.plot([sum(x < 1e-5) for x in ev_all], label="x < 1e-5", color=default_colors[0])

        ax.set_title("zero ev count")
        ax.legend()

    if False:
        ax = axs[i]
        i += 1

        ax.plot([sum(x == 0) for x in ev_all], label="== 0", color=default_colors[0])

        ax.set_title("zero ev count")
        ax.legend()

    if True:
        ax = axs[i]
        i += 1

        ax.plot([sum(x < 0) for x in ev_all], label="< 0", color=default_colors[0])

        #ax.set_title("zero ev count")
        ax.legend()

    if False:
        ax = axs[i]
        i += 1

        ax.plot([sum((x > 0) & (x <= 1e-8)) for x in ev_all], label="0 < x <= 1e-8", color=default_colors[0])

        #ax.set_title("zero ev count")
        ax.legend()

    if False:
        ax = axs[i]
        i += 1

        ax.plot([sum(x < -1e-8) for x in ev_all], label="< -1e-8", color=default_colors[0])

        #ax.set_title("zero ev count")
        ax.legend()

    if True:
        ax = axs[i]
        i += 1

        #ax.plot([sum((1e-6 <= x) & (x <= 1e2)) for x in ev_all], label="1e-8 <= x <= 1e1", color=default_colors[0])

        #ax.set_title("zero ev count")
        #ax.legend()

        ev_all = np.concatenate(log.marg_ev[3:])
        ev_all = ev_all[ev_all < 1e3]
        num = len(log.marg_ev[3:])

        ax.hist(
            ev_all,
            bins=[
                -1e2,
                -1e1,
                -1e0,
                -1e-1,
                -1e-2,
                -1e-3,
                -1e-4,
                -1e-5,
                -1e-6,
                #-1e-7,
                #-1e-8,
                #-1e-9,
                #-1e-10,
                0,
                #1e-10,
                #1e-9,
                #1e-8,
                #1e-7,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1e-0,
                1e1,
                1e2,
                1e3,
                #1e4,
                #1e5,
                #1e6,
                #1e7,
                #1e8,
                #1e9,
                #1e10,
                #1e11
            ])
        ax.set_xscale("symlog", linthresh=1e-6)
        y_vals = ax.get_yticks()
        ax.set_yticklabels(['{:.1f}'.format(x / num) for x in y_vals])
        ax.set_title("hist of all ev < 1e3 (count normalized by num marg-priors)")

    if args.save:
        plt.savefig(args.save)

    if not args.no_gui:
        plt.show()


def main():
    parser = argparse.ArgumentParser("Load multiple PBA logs and plot combined results for comparison.")
    parser.add_argument("path", help="log file path")
    parser.add_argument("--no-gui", action="store_true", help="show plots")
    parser.add_argument("--save", help="save plots to specified file")

    args = parser.parse_args()

    plot(args)


if __name__ == "__main__":
    main()
