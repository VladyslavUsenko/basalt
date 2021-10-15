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
import math

import numpy as np

from pylatex import Subsection, Tabular, TextColor
from pylatex import MultiRow, FootnoteText
from pylatex.utils import italic, bold, NoEscape, escape_latex, dumps_list

from .containers import ExperimentsTable
from .util import format_ratio_percent
from .util import best_two_non_repeating


class ResultsTable(ExperimentsTable):

    def __init__(self, exps, spec, show_values_failed_runs, seq_displayname_mapping, export_basepath):
        super().__init__(exps, spec, show_values_failed_runs, seq_displayname_mapping, export_basepath)

        self.doit()

    def doit(self):

        is_multirow = self.num_metrics > 1 and self.spec.multirow

        def render_metric(value, best, second, decimals, format_string, highlight_top, relative_to):
            if isinstance(value, numbers.Number):
                if relative_to is None or relative_to == 0 or not np.isfinite(relative_to):
                    # absolute number
                    rendered = format_string.format(value, prec=decimals)
                else:
                    # percent
                    rendered = format_ratio_percent(value, relative_to, decimals=decimals)
                if highlight_top:
                    if value == best:
                        rendered = bold(rendered)
                    elif value == second:
                        rendered = italic(rendered)
                return rendered
            else:
                return value

        if self.spec.export_latex:
            row_height = None
        else:
            row_height = 0.65 if is_multirow and self.num_metrics >= 3 else 1

        column_spec = '|r' if self.spec.vertical_bars else 'r'
        t = Tabular('l' + column_spec * self.num_exps, row_height=row_height, pos=['t'])
        escape_header_fun = lambda text: text if self.spec.escape_latex_header else NoEscape(text)
        if self.spec.rotate_header:
            t.add_row([''] + [
                NoEscape(r"\rotatebox{90}{%s}" % escape_latex(escape_header_fun(s.display_name(self.exps[s.name]))))
                for s in self.experiment_specs
            ])
        else:
            t.add_row([''] + [escape_header_fun(s.display_name(self.exps[s.name])) for s in self.experiment_specs])
        t.add_hline()

        for seq in self.seq_names:
            fails = [self.is_failed(self.exps[s.name], seq) for s in self.experiment_specs]
            failure_strings = [self.render_failure(self.exps[s.name], seq) for s in self.experiment_specs]
            values = np.array([self.get_metrics(self.exps[s.name], seq, s.it) for s in self.experiment_specs])

            top_values = list(range(self.num_metrics))
            for c, m in enumerate(self.metrics):
                try:
                    values[:, c] = np.around(values[:, c], m.decimals)
                except IndexError:
                    pass
                non_excluded_values = np.array(values[:, c])
                for i in m.exclude_columns_highlight:
                    non_excluded_values[i] = math.nan
                top_values[c] = best_two_non_repeating(non_excluded_values, reverse=m.larger_is_better)

            if is_multirow:
                rows = [[MultiRow(self.num_metrics, data=self.seq_displayname(seq))]
                       ] + [list(['']) for _ in range(1, self.num_metrics)]
            else:
                rows = [[self.seq_displayname(seq)]]
            for c, (fail, failure_str, value_col) in enumerate(zip(fails, failure_strings, values)):
                if failure_str is not None:
                    if self.spec.color_failed:
                        failure_str = TextColor(self.spec.color_failed, failure_str)
                    if is_multirow:
                        rows[0].append(MultiRow(self.num_metrics, data=failure_str))
                        for r in range(1, self.num_metrics):
                            rows[r].append('')
                    else:
                        rows[0].append(failure_str)
                else:
                    tmp_data = [None] * self.num_metrics
                    for r, m in enumerate(self.metrics):
                        if m.failed_threshold and value_col[r] > m.failed_threshold:
                            obj = "x"
                            if self.spec.color_failed:
                                obj = TextColor(self.spec.color_failed, obj)
                        else:
                            relative_to = None
                            if m.relative_to_column is not None and m.relative_to_column != c:
                                relative_to = values[m.relative_to_column, r]
                            obj = render_metric(value_col[r],
                                                top_values[r][0],
                                                top_values[r][1],
                                                m.effective_display_decimals(),
                                                m.format_string,
                                                m.highlight_top,
                                                relative_to=relative_to)
                            if fail and self.spec.color_failed:
                                obj = TextColor(self.spec.color_failed, obj)
                        tmp_data[r] = obj
                    if self.num_metrics == 1 or is_multirow:
                        for r, obj in enumerate(tmp_data):
                            rows[r].append(obj)
                    else:
                        entry = []
                        for v in tmp_data:
                            entry.append(v)
                            entry.append(NoEscape("~/~"))
                        entry.pop()
                        rows[0].append(dumps_list(entry))

            for row in rows:
                t.add_row(row)

            if is_multirow:
                t.add_hline()

        if self.spec.export_latex:
            os.makedirs(self.export_basepath, exist_ok=True)
            t.generate_tex(os.path.join(self.export_basepath, self.spec.export_latex))

        with self.create(Subsection(self.spec.name, numbering=False)) as p:

            if self.spec.metrics_legend:
                legend = Tabular('|c|', row_height=row_height, pos=['t'])
                legend.add_hline()
                legend.add_row(["Metrics"])
                legend.add_hline()
                for m in self.metrics:
                    legend.add_row([m.display_name])
                legend.add_hline()

                tab = Tabular("ll")
                tab.add_row([t, legend])
                content = tab
            else:
                content = t

            if True:
                content = FootnoteText(content)

            p.append(content)
