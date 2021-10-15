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

from pylatex import Package
from pylatex.base_classes import Container

from ..metric import metrics_from_config
from ..metric import ExperimentSpec
from ..util import alphanum


class MyContainer(Container):

    def __init__(self):
        super().__init__()
        # add packages that seem to not propagate properly from added elements
        self.packages.add(Package("xcolor"))
        self.packages.add(Package('graphicx'))

    def dumps(self):
        return self.dumps_content()


class ExperimentsContainer(MyContainer):

    def __init__(self, seq_displayname_mapping):
        super().__init__()

        self.seq_displayname_mapping = seq_displayname_mapping

    def seq_displayname(self, seq):
        return self.seq_displayname_mapping.get(seq, seq)


class ExperimentsTable(ExperimentsContainer):

    def __init__(self, exps, spec, show_values_failed_runs, seq_displayname_mapping, export_basepath):
        super().__init__(seq_displayname_mapping)
        self.exps = exps
        self.spec = spec
        self.show_values_failed_runs = show_values_failed_runs
        self.export_basepath = export_basepath

        self.experiment_specs = [ExperimentSpec(s) for s in self.spec.experiments]
        self.metrics = metrics_from_config(self.spec.metrics)

        self.seq_names = self.sequence_names([s.name for s in self.experiment_specs])
        self.num_seqs = len(self.seq_names)
        self.num_metrics = len(self.metrics)
        self.num_exps = len(self.experiment_specs)

    def sequence_names(self, experiment_names):
        seq_names = set()
        for s in experiment_names:
            seq_names.update(self.exps[s].sequences(filter_regex=self.spec.filter_regex))

        return sorted(seq_names, key=alphanum)

    def is_failed(self, exp, seq):
        if seq not in exp.runs:
            return True
        return exp.runs[seq].is_failed()

    def render_failure(self, exp, seq):
        if seq in self.spec.override_as_failed:
            return "x"

        if seq not in exp.runs:
            return '?'
        run = exp.runs[seq]

        treat_as_failed = (run.log is None) if self.show_values_failed_runs else run.is_failed()

        if treat_as_failed:
            return run.failure_str()
        else:
            return None

    def get_metrics(self, exp, seq, it):
        if seq not in exp.runs:
            return [math.nan for _ in self.metrics]
        run = exp.runs[seq]

        treat_as_failed = (run.log is None) if self.show_values_failed_runs else run.is_failed()

        if treat_as_failed:
            return [math.nan for _ in self.metrics]

        return [m.get_value(self.exps, exp, seq, it) for m in self.metrics]
        # try:
        #     return [m.get_value(self.exps, exp, seq, it) for m in self.metrics]
        # except AttributeError as e:
        #     if e.args[0].startswith("local_error"):
        #         if not has_imported_sophus():
        #             print("To use local-error, you need to install sophuspy and flush the cache.")
        #             sys.exit(1)
        #         if not exp.runs[seq].log.has_cam_pos:
        #             print("You cannot use local-error for experiment {}, which has no camera positions in the log.".
        #                   format(exp.name))
        #             sys.exit(1)
        #     raise
