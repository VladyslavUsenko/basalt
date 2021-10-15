#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import numpy as np
import os
import math
import functools

import matplotlib

matplotlib.use('Agg')  # Not to use X server. For TravisCI.
import matplotlib.pyplot as plt  # noqa
from matplotlib.ticker import MaxNLocator

prop_cycle = plt.rcParams['axes.prop_cycle']

#default_cycler = (cycler(linestyle=['-', '--', ':', '-.']) *
#                  cycler(color=prop_cycle.by_key()['color']))


class ModulusList(list):

    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        return list.__getitem__(self, key % len(self))


default_colors_finite = prop_cycle.by_key()['color']
default_colors_finite[0] = prop_cycle.by_key()['color'][0]
default_colors_finite[1] = prop_cycle.by_key()['color'][2]
default_colors_finite[2] = prop_cycle.by_key()['color'][3]
default_colors_finite[3] = prop_cycle.by_key()['color'][1]

default_colors = ModulusList(default_colors_finite)
#default_lines = ModulusList(["-", "-", ":", "--", "-.", ":", "--", "-."])
#default_markers = ModulusList(["o", "s", "^", "X", "D", "P", "v", "h"])
default_lines = ModulusList([":", "-", "-.", "--", ":", "--", "-.", "-"])
default_markers = ModulusList(["o", "s", "^", "X", "D", "P", "v", "h"])

from collections import deque
from collections import defaultdict

from pylatex import Figure
from pylatex.utils import NoEscape

from .containers import ExperimentsContainer
from .util import rotation2d


class NoFloatFigure(Figure):
    pass


class Plot(ExperimentsContainer):

    def __init__(self, exps, spec, seq_displayname_mapping, export_basepath):
        super().__init__(seq_displayname_mapping)

        self.width = None

        plotters = dict(nullspace=self.plot_nullspace,
                        eigenvalues=self.plot_eigenvalues,
                        trajectory=self.plot_trajectory)

        plot_fn = plotters[spec.type]
        plot_fn(exps, spec)

        if spec.width is not None:
            self.width = spec.width
        elif self.width is None:
            self.width = 1

        plt.tight_layout()

        saved_file = self._save_plot(spec, export_basepath)

        if "sequence" in spec:
            plot_name = '{} {} {}'.format(spec.type, spec.name, spec.sequence).replace("_", " ")
        else:
            plot_name = '{} {}'.format(spec.type, spec.name).replace("_", " ")

        #with self.create(Subsection(spec.name, numbering=False)) as p:
        with self.create(NoFloatFigure()) as f:
            f.add_image(os.path.abspath(saved_file), width=NoEscape(r'{}\textwidth'.format(self.width)))
            f.add_caption(plot_name)

        # cleanup
        plt.close('all')

    def plot_nullspace(self, exps, spec):

        logs = [exps[e].runs[spec.sequence].log for e in spec.experiments]
        names = [exps[e].display_name for e in spec.experiments]

        num_plots = len(names)

        if num_plots == 4:
            if True:
                if spec.figsize is None:
                    spec.figsize = [10, 2.5]
                fig, axs = plt.subplots(1, 4, figsize=spec.figsize, sharey=True)
            else:
                if spec.figsize is None:
                    spec.figsize = [10, 4.7]
                fig, axs = plt.subplots(2, 2, figsize=spec.figsize, sharey=True)
                axs = axs.flatten()
        else:
            if spec.figsize is None:
                spec.figsize = [6, 2 * num_plots]
            fig, axs = plt.subplots(num_plots, 1, figsize=spec.figsize, sharey=True)

        if num_plots == 1:
            axs = [axs]

        for i, (log, name) in enumerate(zip(logs, names)):

            if log is None:
                continue

            ax = axs[i]

            ns = log.sums.marg_ns[1:]  # skip first prior, which just is all 0
            ns = np.abs(ns)  # cost change may be negative, we are only interested in the norm
            ns = np.maximum(ns, 1e-20)  # clamp at very small value

            markerfacecolor = "white"

            markevery = 1000
            if spec.sequence == "kitti10":
                markevery = 100

            ax.semilogy(
                ns[:, 0],
                ":",
                # label="x",
                color="tab:blue")
            ax.semilogy(
                ns[:, 1],
                ":",
                # label="y",
                color="tab:blue")
            ax.semilogy(
                ns[:, 2],
                ":",
                # label="z",
                label="x, y, z",
                color="tab:blue",
                marker="o",
                markerfacecolor=markerfacecolor,
                markevery=(markevery // 2, markevery))

            ax.semilogy(
                ns[:, 3],
                ":",
                # label="roll",
                color="tab:orange")
            ax.semilogy(
                ns[:, 4],
                ":",
                # label="pitch",
                label="roll, pitch",
                color="tab:orange",
                marker="s",
                markerfacecolor=markerfacecolor,
                markevery=(markevery // 2, markevery))

            ax.semilogy(ns[:, 5],
                        ":",
                        label="yaw",
                        color="tab:green",
                        marker="^",
                        markerfacecolor=markerfacecolor,
                        markevery=(0, markevery))

            ax.semilogy(ns[:, 6],
                        ":",
                        label="random",
                        color="tab:red",
                        marker="D",
                        markerfacecolor=markerfacecolor,
                        markevery=(0, markevery))

            # marker on top of lines;

            ax.semilogy(ns[:, 2],
                        color="None",
                        marker="o",
                        markerfacecolor=markerfacecolor,
                        markeredgecolor="tab:blue",
                        markevery=(markevery // 2, markevery))
            ax.semilogy(ns[:, 4],
                        color="None",
                        marker="s",
                        markerfacecolor=markerfacecolor,
                        markeredgecolor="tab:orange",
                        markevery=(markevery // 2, markevery))

            #ax.set_yscale("symlog", linthresh=1e-12)

            ax.set_title(name)

            ax.set_yticks([1e-17, 1e-12, 1e-7, 1e-2, 1e3, 1e8])

            if spec.sequence == "kitti10":
                ax.set_xticks([i * 100 for i in range(4)])
                #ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([i * 100 + 50 for i in range(4)]))

            if i == 0:
                ax.set_ylabel("$\\Delta E_m$", rotation=0)
                ax.yaxis.set_label_coords(-0.05, 1.05)

            if i == num_plots - 1:
                ax.legend(loc=spec.legend_loc)

            if spec.ylim.top is not None:
                ax.set_ylim(top=spec.ylim.top)
            if spec.ylim.bottom is not None:
                ax.set_ylim(bottom=spec.ylim.bottom)

        if spec.suptitle:
            fig.suptitle(spec.suptitle)

    def plot_eigenvalues(self, exps, spec):

        logs = [exps[e].runs[spec.sequence].log for e in spec.experiments]
        names = [exps[e].display_name for e in spec.experiments]

        num_plots = 1

        if spec.figsize is None:
            spec.figsize = [5.2, 2 * num_plots]

        fig, axs = plt.subplots(num_plots, 1, figsize=spec.figsize)

        ax = axs

        for i, (log, name) in enumerate(zip(logs, names)):
            if log is not None:
                min_ev = [np.min(e) for e in log.sums.marg_ev[1:]]
                #ax.plot(min_ev, ":", label=name, color=default_colors[i])
                ax.plot(min_ev, default_lines[i], label=name, color=default_colors[i])

        ax.set_yscale("symlog", linthresh=1e-8)
        ax.legend(loc=spec.legend_loc)

        #ax.set_title("smallest eigenvalue {} {}".format(name, spec.sequence))

        if spec.sequence == "eurocMH01":
            ax.set_xticks([i * 1000 for i in range(4)])
            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([i * 1000 + 500 for i in range(4)]))

        if spec.sequence == "kitti10":
            ax.set_xticks([i * 100 for i in range(4)])
            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator([i * 100 + 50 for i in range(4)]))

        ax.set_yticks([-1e4, -1e-4, 0.0, 1e-4, 1e4])
        ax.set_ylim(bottom=-1e8, top=1e8)
        # ax.yaxis.tick_right()
        ax.set_ylabel("$\\sigma_{min}$", rotation=0)
        ax.yaxis.set_label_coords(0, 1.05)

        if spec.ylim.top is not None:
            ax.set_ylim(top=spec.ylim.top)
        if spec.ylim.bottom is not None:
            ax.set_ylim(bottom=spec.ylim.bottom)

        if spec.suptitle:
            fig.suptitle(spec.suptitle)

    def plot_trajectory(self, exps, spec):

        #self.width = 1.5

        runs = [exps[e].runs[spec.sequence] for e in spec.experiments]
        names = [exps[e].display_name for e in spec.experiments]

        linewidth_factor = 3

        R = rotation2d(spec.rotate2d)

        traj_axes_idx = self._axes_spec_to_index(spec.trajectory_axes)

        if spec.figsize is None:
            spec.figsize = [6.4, 4.8]

        fig, ax = plt.subplots(figsize=spec.figsize)

        ax.axis("equal")
        ax.axis('off')
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")

        gt_color = "tab:grey"
        #gt_color = "black"

        # take gt-trajectory from first experiment:
        if runs[0].traj_gt is not None:
            gt = runs[0].traj_gt[:, traj_axes_idx].T
            gt = np.matmul(R, gt)
            ax.plot(gt[0, :],
                    gt[1, :],
                    '-',
                    zorder=1,
                    linewidth=1 * linewidth_factor,
                    color=gt_color,
                    label="ground truth")

        # https://matplotlib.org/stable/gallery/color/named_colors.html
        linestyles = [":", ":", "--", "-"]
        colors = [default_colors[1], default_colors[3]]
        #colors = ["tab:blue", "tab:orange"]
        linewidths = [2, 1]

        for i, (r, name) in enumerate(zip(runs, names)):
            # plot in decreasing zorder
            #zorder = len(runs) - i + 1

            zorder = i + 2

            if r.traj_est is not None:
                pos = r.traj_est[:, traj_axes_idx].T
                pos = np.matmul(R, pos)
                ax.plot(
                    pos[0, :],
                    pos[1, :],
                    linestyles[i],
                    #default_lines[i],
                    zorder=zorder,
                    linewidth=linewidths[i] * linewidth_factor,
                    label=name,
                    color=colors[i])

        #ax.set_xlim(np.min(x_gt), np.max(x_gt))
        #ax.set_ylim(np.min(y_gt), np.max(y_gt))

        #lines = [gt]
        #colors = ['black']
        #line_segments = LineCollection(lines, colors=colors, linestyle='solid')

        #ax.add_collection(line_segments)

        ax.legend(loc=spec.legend_loc)

        if spec.title is not None:
            ax.set_title(spec.title.format(sequence=self.seq_displayname(spec.sequence)))

    @staticmethod
    def _axes_spec_to_index(axes_spec):
        index = []
        assert len(axes_spec) == 2, "Inalid axes_spec {}".format(axes_spec)
        for c in axes_spec:
            if c == "x":
                index.append(0)
            elif c == "y":
                index.append(1)
            elif c == "z":
                index.append(2)
            else:
                assert False, "Inalid axes_spec {}".format(axes_spec)
        return index

    # static:
    filename_counters = defaultdict(int)

    def _save_plot(self, spec, basepath, extension=".pdf"):

        os.makedirs(basepath, exist_ok=True)

        if "sequence" in spec:
            filename = '{}_{}_{}'.format(spec.type, spec.name, spec.sequence)
        else:
            filename = '{}_{}'.format(spec.type, spec.name)

        filename = filename.replace(" ", "_").replace("/", "_")

        Plot.filename_counters[filename] += 1
        counter = Plot.filename_counters[filename]
        if counter > 1:
            filename = "{}-{}".format(filename, counter)

        filepath = os.path.join(basepath, "{}.{}".format(filename, extension.strip('.')))

        plt.savefig(filepath)

        return filepath
