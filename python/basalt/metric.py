#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import copy

import numpy as np


class ExperimentSpec:

    def __init__(self, string):
        if "@it" in string:
            self.name, it = string.split("@it")
            self.it = int(it)
        else:
            self.name = string
            self.it = -1

    def display_name(self, exp):
        if self.it == -1:
            return exp.display_name
        else:
            return "{} @ it{}".format(exp.display_name, self.it)


class Metric:

    def __init__(self,
                 display_name,
                 accessor,
                 decimals,
                 format_string="{:.{prec}f}",
                 highlight_top=True,
                 geometric_mean=False,
                 larger_is_better=False):
        self.display_name = display_name
        self.accessor = accessor
        self.decimals = decimals
        self.display_decimals = None
        self.relative_to_column = None
        self.relative_to_experiment = None
        self.relative_to_metric = None
        self.ratio = True
        self.format_string = format_string
        self.highlight_top = highlight_top
        self.larger_is_better = larger_is_better
        self.exclude_columns_highlight = []
        self.geometric_mean = geometric_mean
        self.failed_threshold = None

    def set_config(self, spec):
        # change defaults in case of "relative_to_..." display mode
        if any(k in spec for k in ["relative_to_column", "relative_to_experiment", "relative_to_metric"]):
            # maybe overwritten by explicit "decimals" / "format_string" below
            self.decimals = 3
            self.display_decimals = 3
            self.format_string = "{:.3f}"
            self.geometric_mean = True

        if "display_name" in spec:
            self.display_name = spec.display_name
        if "decimals" in spec:
            self.decimals = spec.decimals
        if "display_decimals" in spec:
            self.display_decimals = spec.display_decimals
        if "relative_to_column" in spec:
            self.relative_to_column = spec.relative_to_column
        if "relative_to_experiment" in spec:
            self.relative_to_experiment = ExperimentSpec(spec.relative_to_experiment)
        if "relative_to_metric" in spec:
            self.relative_to_metric = spec.relative_to_metric
        if "ratio" in spec:
            self.ratio = spec.ratio
        if "format_string" in spec:
            self.format_string = spec.format_string
        if "highlight_top" in spec:
            self.highlight_top = spec.highlight_top
        if "larger_is_better" in spec:
            self.larger_is_better = spec.larger_is_better
        if "exclude_columns_highlight" in spec:
            self.exclude_columns_highlight = spec.exclude_columns_highlight
        if "geometric_mean" in spec:
            self.geometric_mean = spec.geometric_mean
        if "failed_threshold" in spec:
            self.failed_threshold = spec.failed_threshold

    def effective_display_decimals(self):
        if self.display_decimals is not None:
            return self.display_decimals
        else:
            return self.decimals

    def get_value(self, exps, e, s, it):
        #try:
        value = self.accessor(e.runs[s].log, it)
        #except AttributeError as err:
        #    raise

        if self.relative_to_metric is not None:
            relative_to_metric_accessor = self.relative_to_metric.accessor
        else:
            relative_to_metric_accessor = self.accessor

        if self.relative_to_experiment is not None:
            relative_to_log = exps[self.relative_to_experiment.name].runs[s].log
            relative_to_it = self.relative_to_experiment.it
        else:
            relative_to_log = e.runs[s].log
            relative_to_it = it

        if self.relative_to_metric is not None or self.relative_to_experiment is not None:
            base_value = relative_to_metric_accessor(relative_to_log, relative_to_it)
            if self.ratio:
                value = value / base_value
            else:
                value = base_value - value
        return value


def peak_memory_opt(l, it):
    if it == -1:
        index = -1
    else:
        index = int(l.all.num_it[:it + 1].sum()) - 1
    return l.all.resident_memory_peak[index] / 1024**2


# yapf: disable
metric_desc = dict(
    ev_min=Metric("min ev", lambda l, it: min(min(x) for x in l.sums.marg_ev), 1),
    avg_num_it=Metric("avg #it", lambda l, it: np.mean(l.sums.num_it), 1),
    avg_num_it_failed=Metric("avg #it-fail", lambda l, it: np.mean(l.sums.num_it_rejected), 1),
    duration=Metric("duration (s)", lambda l, it: l.duration(), 1),
    time_marg=Metric("t marg", lambda l, it: np.sum(l.sums.marginalize), 2),
    time_opt=Metric("t opt", lambda l, it: np.sum(l.sums.optimize), 2),
    time_optmarg=Metric("t opt", lambda l, it: np.sum(l.sums.optimize) + np.sum(l.sums.marginalize), 2),
    time_exec=Metric("t exec", lambda l, it: l.vio.exec_time_s[0], 1),
    time_exec_realtimefactor=Metric("t exec (rtf)", lambda l, it: l.duration() / l.vio.exec_time_s[0], 1, larger_is_better=True),
    time_measure=Metric("t meas", lambda l, it: np.sum(l.sums.measure), 1),
    time_measure_realtimefactor=Metric("t meas (rtf)", lambda l, it: l.duration() / np.sum(l.sums.measure), 1, larger_is_better=True),
    time_exec_minus_measure=Metric("t exec - meas", lambda l, it: l.vio.exec_time_s[0] - np.sum(l.sums.measure), 1),
    time_measure_minus_optmarg=Metric("t exec - (opt + marg)", lambda l, it: np.sum(l.sums.measure) - (np.sum(l.sums.optimize) + np.sum(l.sums.marginalize)), 1),
    ate_num_kfs=Metric("ATE #kf", lambda l, it: l.vio.ate_num_kfs[0], 0),
    ate_rmse=Metric("ATE", lambda l, it: l.vio.ate_rmse[0], 3),
    peak_memory=Metric("mem peak (MB)", lambda l, it: l.vio.resident_memory_peak[0] / 1024**2, 1),
    #peak_memory_opt=Metric("mem peak opt (MB)", lambda l, it: l.all.resident_memory_peak[l.all.num_it[:it].sum()-1] / 1024**2, 1),
    peak_memory_opt=Metric("mem peak opt (MB)", peak_memory_opt, 1),
)
# yapf: enable


def metrics_from_config(spec):

    def get_from_spec(m):
        if isinstance(m, str):
            obj = copy.copy(metric_desc[m])
        else:
            obj = copy.copy(metric_desc[m.name])
            obj.set_config(m)
        if obj.relative_to_metric is not None:
            obj.relative_to_metric = get_from_spec(obj.relative_to_metric)

        return obj

    return [get_from_spec(m) for m in spec]
