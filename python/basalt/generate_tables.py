#
# BSD 3-Clause License
#
# This file is part of the Basalt project.
# https://gitlab.com/VladyslavUsenko/basalt.git
#
# Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
# All rights reserved.
#
import argparse
import os

from pylatex import Document, Section, Package, NewPage
from pylatex import Command
from pylatex.base_classes import Arguments

from .experiments import load_experiments_config
from .experiments import Experiment
from .latex.templates import screenread_sty
from .util import os_open_file
from .latex.results_table import ResultsTable
from .latex.summarize_sequences_table import SummarizeSequencesTable
from .latex.plot import Plot


def generate_tables(args):

    if args.change_directory:
        os.chdir(args.change_directory)

    config = load_experiments_config(args.config, args)

    exps = Experiment.load_all(config.experiments,
                               config_file=args.config,
                               base_path=config.options.base_path,
                               cache_dir=config.options.cache_dir,
                               seq_name_mapping=config.seq_name_mapping)

    doc = Document(geometry_options={"tmargin": "1cm", "lmargin": "1cm"})

    export_basepath = "{}-export".format(config.options.output_path)

    curr = doc

    hide_all = False
    for spec in config.results:
        if spec.show:

            if spec["class"] == "section":
                if spec.clearpage:
                    curr.append(NewPage())
                if spec.pagewidth:
                    curr.append(Command("SetPageScreenWidth", Arguments(spec.pagewidth)))
                else:
                    curr.append(Command("RestorePageScreenWidth"))
                hide_all = False
                curr = Section(spec.name)
                doc.append(curr)
                continue

            if hide_all:
                continue

            if spec.clearpage:
                curr.append(NewPage())

            elif spec["class"] == "results_table":
                elem = ResultsTable(exps,
                                    spec,
                                    show_values_failed_runs=config.options.show_values_failed_runs,
                                    seq_displayname_mapping=config.seq_displayname_mapping,
                                    export_basepath=export_basepath)
            elif spec["class"] == "summarize_sequences_table":
                elem = SummarizeSequencesTable(exps,
                                               spec,
                                               show_values_failed_runs=config.options.show_values_failed_runs,
                                               seq_displayname_mapping=config.seq_displayname_mapping,
                                               export_basepath=export_basepath)
            elif spec["class"] == "plot":
                elem = Plot(exps,
                            spec,
                            seq_displayname_mapping=config.seq_displayname_mapping,
                            export_basepath=export_basepath)
            else:
                raise RuntimeError("Invalid results class {}".format(spec["class"]))

            curr.append(elem)
        else:
            if spec["class"] == "section":
                hide_all = True
                continue

    # generate auxiliary tex files
    if config.options.screenread:
        output_dir = os.path.dirname(config.options.output_path)
        screenread_path = output_dir + "/screenread.sty"
        with open(screenread_path, "w") as f:
            f.write(screenread_sty)
        doc.packages.add(Package('screenread'))

    # create nofloatfigure environment
    doc.preamble.append(
        Command("newenvironment", Arguments("nofloatfigure", Command("captionsetup", Arguments(type="figure")), "")))
    doc.packages.add(Package('caption'))
    doc.packages.add(Package('mathtools'))

    # render latex
    doc.generate_pdf(config.options.output_path, clean_tex=not args.dont_clean_tex)

    # cleanup
    if config.options.screenread and not args.dont_clean_tex:
        os.remove(screenread_path)

    # open the generated pdf
    if args.open:
        os_open_file(config.options.output_path + ".pdf")


def main():

    parser = argparse.ArgumentParser("Load basalt experiment logs and generate result tables and plots.")
    parser.add_argument("-C",
                        "--change-directory",
                        default=None,
                        help="Change directory to this folder before doing anything else")
    parser.add_argument("--config", default="experiments.toml", help="specs for experiments to load")
    parser.add_argument("--base-path", default=None, help="overwrite basepath for loading logs defined in the config")
    parser.add_argument("--output-path", default=None, help="output filepath")
    parser.add_argument("--dont-clean-tex", action="store_true", help="don't remove tex file after generation")
    parser.add_argument("--cache-dir", default=None, help="load/save experiments cache from/to give folder")
    parser.add_argument("--overwrite-cache", action="store_true", help="reload all experiments independent of cache")
    parser.add_argument("--dont-show-values-failed-runs",
                        action="store_true",
                        help="don't attempt to show values for failed logs based on partial logs")
    parser.add_argument("--open", action="store_true", help="open after generation")

    args = parser.parse_args()

    generate_tables(args)


if __name__ == "__main__":
    main()
