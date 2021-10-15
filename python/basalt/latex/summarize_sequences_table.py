#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import numbers
import os
import scipy.stats
import numpy as np

from pylatex import Subsection, FootnoteText, Tabular, NoEscape, escape_latex
from pylatex.utils import italic, bold

from .containers import ExperimentsTable
from .util import best_two_non_repeating


class SummarizeSequencesTable(ExperimentsTable):

    def __init__(self, exps, spec, show_values_failed_runs, seq_displayname_mapping, export_basepath):
        super().__init__(exps, spec, show_values_failed_runs, seq_displayname_mapping, export_basepath)

        self.doit()

    def doit(self):

        def render_metric(value, best, second, decimals, format_string):
            if isinstance(value, numbers.Number):
                rendered = format_string.format(value, prec=decimals)
                if value == best:
                    rendered = bold(rendered)
                elif value == second:
                    rendered = italic(rendered)
                return rendered
            else:
                return value

        values = np.empty((self.num_metrics, self.num_seqs, self.num_exps))

        for i, seq in enumerate(self.seq_names):
            for j, s in enumerate(self.experiment_specs):
                values[:, i, j] = np.array(self.get_metrics(self.exps[s.name], seq, s.it))

        means = np.empty((self.num_metrics, self.num_exps))
        for i, m in enumerate(self.metrics):
            if m.geometric_mean:
                means[i, :] = scipy.stats.gmean(values[i, :, :], axis=0)
            else:
                means[i, :] = np.mean(values[i, :, :], axis=0)

        t = Tabular('l' + 'c' * self.num_exps)

        t.add_hline()
        escape_header_fun = lambda text: text if self.spec.escape_latex_header else NoEscape(text)
        if self.spec.rotate_header:
            t.add_row([self.spec.header] + [
                NoEscape(r"\rotatebox{90}{%s}" % escape_latex(escape_header_fun(s.display_name(self.exps[s.name]))))
                for s in self.experiment_specs
            ])
        else:
            t.add_row([self.spec.header] +
                      [escape_header_fun(s.display_name(self.exps[s.name])) for s in self.experiment_specs])
        t.add_hline()

        for i, m in enumerate(self.metrics):
            row_values = np.around(means[i, :], m.decimals)
            top_values = best_two_non_repeating(row_values, reverse=m.larger_is_better)
            row = [m.display_name]
            for v in row_values:
                # TODO: use NoEscape only if certain flag is enabled?
                row.append(
                    NoEscape(
                        render_metric(v, top_values[0], top_values[1], m.effective_display_decimals(),
                                      m.format_string)))
            t.add_row(row)

        t.add_hline()

        if self.spec.export_latex:
            os.makedirs(self.export_basepath, exist_ok=True)
            t.generate_tex(os.path.join(self.export_basepath, self.spec.export_latex))

        with self.create(Subsection(self.spec.name, numbering=False)) as p:
            p.append(FootnoteText(t))
