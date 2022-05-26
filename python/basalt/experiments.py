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
import re
import json
import hashlib
import pickle
import toml
import argparse

from string import Template
from glob import glob
from collections.abc import Mapping
from munch import Munch
from munch import munchify
from copy import deepcopy
from collections import abc

from .run import Run

from .util import copy_subdict

_CURRENT_CACHE_VERSION = '1.3'
"""cache version that can be incremented to invalidate all cache files in case the format changes"""


def version_less(vstr1, vstr2):
    """Order for sorting versions in the format a.b.c"""
    return vstr1.split(".") < vstr2.split(".")


def compute_caching_hash(d):
    """Generate a hash from a dictionary to use as a cache file name

    This is intended to be used for experiments cache files
    """
    string = json.dumps(d, sort_keys=True, ensure_ascii=True)
    h = hashlib.sha1()
    h.update(string.encode('utf8'))
    return h.hexdigest()


class Experiment:
    """Holds the logs for one experiment: a single odometry config run on a set of sequences

    For one experiment, each sequence may have at most one Run.

    Since for each run we have multiple log files and there may be many runs, we
    cache the loaded configs / output / log files (after preprocessing) into a single
    binary cache file (pickle). This significantly speeds up loading results when
    we have many experiments defined in a single experiments config file.
    """

    def __init__(self,
                 log_dirs,
                 name,
                 display_name=None,
                 description=None,
                 caching_hash=None,
                 spec=None,
                 seq_name_mapping=None,
                 extend=None,
                 extend_override=False):
        """Load an experiment and all it's runs from a set of directories

        There may be no duplicate runs of the same sequence.

        :param log_dirs: list of directories to look for runs in
        :param name: experiment name
        :param display_name: optional experiment display name
        :param description: optional experiment description
        :param caching_hash: own caching hash; mostly used to combine this hash with the has of extending experiments
        :param spec: the config spec for this experiment; mainly informational for informative error messages; the
               functionally relevant information has already be extracted an preprocessed (other arguments)
        :param seq_name_mapping: optional mapping of sequence names; may contain only part of the sequences
        :param extend: optionally provide base experiment whose runs are copied (and possibly extended)
        :param extend_override: if True, sequences in the extended experiment may be replaced, if they are also found in `log_dirs`
        """

        self.name = name
        self.display_name = display_name
        self.description = description
        self.caching_hash = caching_hash
        self.spec = spec
        self.runs = dict()

        if extend is not None:
            for k, v in extend.runs.items():
                self.runs[k] = deepcopy(v)

        seqs_ok_to_override = set(self.runs.keys()) if extend_override else set()

        for d in log_dirs:
            run = Run(d, seq_name_mapping)
            if run.seq_name in self.runs:
                if run.seq_name in seqs_ok_to_override:
                    seqs_ok_to_override.remove(run.seq_name)  # ok only once
                else:
                    if extend is not None and run.seq_name in extend.runs and not extend_override:
                        raise RuntimeError(
                            str.format(
                                "{} appears both in the extended experiment {} and in the extending "
                                "experiment {} but extend_override is False:\n - {}\n - {}\n", run.seq_name,
                                extend.name, self.name, extend.runs[run.seq_name].dirpath, run.dirpath))
                    else:
                        raise RuntimeError(
                            str.format(
                                "{} appears multiple times in experiment {}:\n - {}\n - {}\n"
                                "Do your experiment pattern(s) '{}' match too many directories? "
                                "Delete the additional runs or narrow the pattern.", run.seq_name, self.name,
                                self.runs[run.seq_name].dirpath, run.dirpath, "', '".join(self.spec["pattern"])))
            self.runs[run.seq_name] = run

    def sequences(self, filter_regex=None):
        """return list of sequence names found for this experiment

        :param filter_regex: if provided, return only they sequences that match the regex
        """
        if filter_regex is None:
            return self.runs.keys()
        else:
            return [k for k in self.runs.keys() if re.search(filter_regex, k)]

    @staticmethod
    def load_spec(spec, base_path, cache_dir, seq_name_mapping=None, extra_filter_regex=None, other_specs=[]):
        """Load a single experiment from logs or cache

        The cache key is determined by the 'pattern', 'filter_regex' and 'extend' keys
        in the spec. That means changing the name or display name for example doesn't
        invalidate the cache. If the experiment is not found in cache, it is loaded from
        the run directories and then saved in cache.

        :param spec: experiment spec from the config file
        :param base_path: base folder to search for run dirs in
        :param cache_dir: cache directory
        :param seq_name_mapping: optional sequence name mapping
        :param extra_filter_regex: additional filter to limit the loaded sequences on top of what is defined in the spec; if set, caching is disabled
        :param other_specs: other experiment specs in case our spec has the 'extend' option defined
        :return: loaded Experiment object
        """

        # disable cache if extra filtering
        if extra_filter_regex is not None:
            cache_dir = None

        # extending some other experiment:
        extend = None
        if "extend" in spec:
            other_spec = next((s for s in other_specs if s.name == spec.extend), None)
            if other_spec is None:
                raise RuntimeError("Experiment {} extends unknown experiment {}.".format(spec.name, spec.extend))
            extend = Experiment.load_spec(other_spec,
                                          base_path,
                                          cache_dir,
                                          seq_name_mapping=seq_name_mapping,
                                          extra_filter_regex=extra_filter_regex,
                                          other_specs=other_specs)

        caching_hash = None
        if cache_dir:
            caching_spec = copy_subdict(spec, ["pattern", "filter_regex"])
            if extend is not None:
                caching_spec["extend"] = extend.caching_hash
            caching_hash = compute_caching_hash(caching_spec)

            cache_filename = "experiment-cache-{}.pickle".format(caching_hash)
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.isfile(cache_path):
                if not spec.overwrite_cache:
                    with open(cache_path, 'rb') as f:
                        cache = pickle.load(f)
                    if cache.version != _CURRENT_CACHE_VERSION:
                        print("> experiment: {} (cache {} has version {}; expected {})".format(
                            spec.name, cache_path, cache.version, _CURRENT_CACHE_VERSION))
                    else:
                        print("> experiment: {} (from cache: {})".format(spec.name, cache_path))
                        exp = cache.experiment
                        # overwrite names according to config
                        exp.name = spec.name
                        exp.display_name = spec.display_name
                        exp.description = spec.description
                        return exp
                else:
                    print("> experiment: {} (overwrite cache: {})".format(spec.name, cache_path))
            else:
                print("> experiment: {} (cache doesn't exist: {})".format(spec.name, cache_path))
        else:
            print("> experiment: {}".format(spec.name))

        log_dirs = Experiment.get_log_dirs(base_path, spec, filter_regex=extra_filter_regex)

        kwargs = copy_subdict(spec, ["name", "display_name", "description", "extend_override"])
        exp = Experiment(log_dirs,
                         caching_hash=caching_hash,
                         seq_name_mapping=seq_name_mapping,
                         extend=extend,
                         spec=deepcopy(spec),
                         **kwargs)

        if cache_dir:
            cache = Munch(version=_CURRENT_CACHE_VERSION, experiment=exp, spec=caching_spec)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
            print("experiment {} -> saved cache {}".format(spec.name, cache_path))

        return exp

    @staticmethod
    def get_log_dirs(base_path, spec, filter_regex=None):
        """Return list of run directories given an experiments spec

        :param base_path: base directory to search in
        :param spec: experiment spec, e.g. from an experiments config file
        :param filter_regex: optional additional regex; limits result to matching paths
        :return: list of (filtered) paths (joined with base path)
        """
        log_dirs = [d for p in spec.pattern for d in glob(os.path.join(base_path, p)) if Run.is_run_dir(d)]
        if spec.filter_regex:
            log_dirs = [d for d in log_dirs if re.search(spec.filter_regex, d)]
        if filter_regex:
            log_dirs = [d for d in log_dirs if re.search(filter_regex, d)]
        return log_dirs

    @staticmethod
    def load_all(specs, config_file, base_path, cache_dir, seq_name_mapping=None):
        """Load a set experiments from log files or cache

        If there is more than one experiment with the same name, an error is raised.

        :param specs: list of experiments specs, e.g. from a experiments config file
        :param config_file: experiments config file path (currently unused)
        :param base_path: base directory relative to which all patterns in experiments are search for
        :param cache_dir: folder to look for and/or save cached experiments
        :param seq_name_mapping: optional mapping of sequence names
        :return: a dict {name: experiment}
        """

        # Note: Seems saving everything to one cache file isn't much faster than the per-experiments cache...
        use_combined_cache = False

        # load all from cache
        if use_combined_cache and cache_dir:

            overwrite_cache_any = any(e.overwrite_cache for e in specs)
            caching_specs = munchify([{k: v for k, v in s.items() if k not in ["overwrite_cache"]} for s in specs])
            meta_info = Munch(version=_CURRENT_CACHE_VERSION, options=Munch(base_path=base_path), specs=caching_specs)

            config_filename = os.path.splitext(os.path.basename(config_file))[0]
            cache_filename = "experiment-cache-{}.pickle".format(config_filename)
            cache_path = os.path.join(cache_dir, cache_filename)

            if os.path.isfile(cache_path):
                if not overwrite_cache_any:
                    with open(cache_path, 'rb') as f:
                        cached_meta_info = pickle.load(f)
                        if cached_meta_info == meta_info:
                            print("> loading from cache: {}".format(cache_path))
                            exps = pickle.load(f)
                            return exps

        # load individually
        exps = dict()
        for spec in specs:
            if spec.name in exps:
                raise RuntimeError("experiment {} is duplicate".format(spec.name))
            exps[spec.name] = Experiment.load_spec(spec,
                                                   base_path,
                                                   cache_dir,
                                                   seq_name_mapping=seq_name_mapping,
                                                   other_specs=specs)

        # save all to cache
        if use_combined_cache and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(meta_info, f)
                pickle.dump(exps, f)
            print("> saved cache {}".format(cache_path))

        return exps


def load_experiments_config(path, args=None):
    """Load experiments config file, applying substitutions and setting defaults

    An experiments config file defines general options, locations of experimental runs,
    and results sections that define tables and plots to render.

    Substitutions and templates can be used to more concisely describe repetitive
    definitions (e.g. generate the same plot for ALL runs of an experiment).

    :param log_dirs: optional command line arguments to override some values in the config
    :type log_dirs: Union[dict, argparse.Namespace]
    """

    config = munchify(toml.load(path))

    # default config:
    config.setdefault("options", Munch())
    config.options.setdefault("base_path", "$config_dir")
    config.options.setdefault("cache_dir", "cache")
    config.options.setdefault("output_path", "results")
    config.options.setdefault("filter_regex", None)
    config.options.setdefault("overwrite_cache", False)
    config.options.setdefault("show_values_failed_runs", True)
    config.options.setdefault("screenread", False)
    config.options.setdefault("import_experiments", [])
    config.setdefault("seq_name_mapping", dict())
    config.setdefault("seq_displayname_mapping", dict())
    config.setdefault("substitutions", [])
    config.setdefault("templates", [])
    config.setdefault("experiments", [])
    config.setdefault("results", [])

    # overrides from command line
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    # values
    for k in ["base_path", "cache_dir", "output_path", "filter_regex"]:
        if k in args and args[k] is not None:
            config.options[k] = args[k]

    # positive flags
    for k in ["overwrite_cache"]:
        if k in args and args[k]:
            config.options[k] = True

    # negative flags
    for k in ["dont_show_values_failed_runs"]:
        if k in args and args[k]:
            config.options[k] = False

    # collapse all substitutions into one dict
    static_subs = dict()
    for d in config.substitutions:
        for k, v in d.items():
            if k in static_subs:
                raise RuntimeError("substitution {} defined multiple times".format(k))
            static_subs[k] = v

    # create dictionary from list of templates (index by name)
    template_definitions = dict()
    for t in config.templates:
        template_definitions[t._name] = t

    # substituion helper
    var_pattern = re.compile(r"\$\{(\w+)\}")  # match '${foo}'

    def substitute(obj, subs):
        if isinstance(obj, Mapping):
            # For mappings in general we simply recurse the 'substitute' call for the dict values.
            # In case the '_template' key is present, we do template expansion.
            if "_template" in obj:
                # template expansion

                # single templates can be abbreviated by not putting them in a list
                # --> put the in list now to make following code the same for either case
                templates = obj._template if isinstance(obj._template, list) else [obj._template]

                # recurse 'substitute' on non-templated part
                prototype = {k: substitute(v, subs) for k, v in obj.items() if not k.startswith("_")}

                # loop over all templates
                result = [Munch()]
                for tmpl in templates:

                    # which arguments are defined?
                    args = [k for k in tmpl if not k.startswith("_")]

                    # check template definition
                    tmpl_def = template_definitions[tmpl._name]
                    tmpl_args = tmpl_def._arguments if "_arguments" in tmpl_def else []
                    if set(args) != set(tmpl_args):
                        raise RuntimeError("Template {} required arguments {}, but supplied {} during expansion".format(
                            tmpl._name, tmpl_def._arguments, args))

                    # apply template definition to all new objects
                    tmp = result
                    result = list()
                    for new_obj in tmp:
                        # create substitutions from template arguments (recursing 'substitute' call)
                        all_argument_combinations = [dict()]  # start with single combination (usual case)
                        for arg in args:
                            if isinstance(tmpl[arg], Mapping) and "_argument" in tmpl[arg]:
                                if tmpl[arg]._argument == "product":
                                    # given list of alternative argument values: create combination for each of them
                                    tmp2 = all_argument_combinations
                                    all_argument_combinations = list()
                                    for d in tmp2:
                                        for val in substitute(tmpl[arg]._value, subs):
                                            d_new = deepcopy(d)
                                            d_new[arg] = val
                                            all_argument_combinations.append(d_new)
                                else:
                                    raise RuntimeError("arugment type {} not for argument {} not implemented".format(
                                        tmpl[arg]._argument, arg))
                            else:
                                # simple argument: append to all combintations
                                for d in all_argument_combinations:
                                    assert (arg not in d)
                                    d[arg] = substitute(tmpl[arg], subs)

                        # for each argument combination, create substitutions and apply template definition
                        for expanded_args in all_argument_combinations:

                            subs_with_args = dict(subs)
                            subs_with_args.update(expanded_args)

                            # merge template definition into result, while recursing substitute call with augmented substitutions
                            new_obj2 = deepcopy(new_obj)
                            for k, v in tmpl_def.items():
                                if not k.startswith("_"):
                                    # later templates can override keys from earlier ones
                                    new_obj2[k] = substitute(deepcopy(v), subs_with_args)

                            result.append(new_obj2)

                # do prototype keys last, since they may override template keys (we already recursed)
                for new_obj in result:
                    for k, v in prototype.items():
                        new_obj[k] = deepcopy(v)

                if len(result) == 1:
                    return new_obj
                else:
                    return Munch(_return="splice", _value=result)
            else:
                # default case
                for k, v in obj.items():
                    obj[k] = substitute(v, subs)
                return obj
        elif isinstance(obj, list):
            # Go over elements of list and recurse the 'substitute' call.
            # In certain cases the returned value can indicate that we should splice in the resulting list instead of
            # just inserting it.
            tmp = list()
            for v in obj:
                val = substitute(v, subs)
                if isinstance(val, dict) and "_return" in val:
                    if val._return == "splice":
                        tmp.extend(val._value)
                    else:
                        raise RuntimeError("Unknown return type {}".format(val._return))
                else:
                    tmp.append(val)
            return tmp
        elif isinstance(obj, str):
            if len(obj) > 2 and obj[0] == "<" and obj[-1] == ">":
                # if string is '<FOO>', the whole string is replaced by the substitution defined for FOO
                var = obj[1:-1]
                if var in subs:
                    return substitute(subs[var], subs)
                else:
                    raise RuntimeError("Unknown substitution <{}>".format(var))
            else:
                # otherwise, find occurances of ${FOO} in the string an replace by a string representation
                # of the substitution defined for FOO
                obj, n = var_pattern.subn(lambda m: str(subs[m.group(1)]), obj)
                if n > 0:
                    # something change --> recurse
                    return substitute(obj, subs)
                else:
                    # no substitution --> just return
                    return obj
        else:
            return obj

    # apply substitutions
    config.experiments = substitute(config.experiments, static_subs)
    config.results = substitute(config.results, static_subs)

    # set default values for experiments specs
    for spec in config.experiments:
        spec.setdefault("display_name", spec.name)
        spec.setdefault("description", None)
        spec.setdefault("filter_regex", config.options.filter_regex)
        spec.setdefault("overwrite_cache", config.options.overwrite_cache)
        spec.setdefault("pattern", [])
        spec.pattern = [spec.pattern] if isinstance(spec.pattern, str) else spec.pattern  # ensure list
        assert isinstance(spec.pattern, abc.Sequence), "pattern {} in experiment {} is neither string nor list".format(
            spec.pattern, spec.name)

    # results: backwards-compatibility -- move old sections into 'results'
    if "results_tables" in config:
        for spec in config.results_tables:
            spec["class"] = "results_table"
            config.results.append(spec)
        del config["results_tables"]

    if "summarize_sequences_tables" in config:
        for spec in config.summarize_sequences_tables:
            spec["class"] = "summarize_sequences_table"
            config.results.append(spec)
        del config["summarize_sequences_tables"]

    if "plots" in config:
        for spec in config.plots:
            spec["class"] = "plot"
            config.results.append(spec)
        del config["plots"]

    if "overview_tables" in config:
        for spec in config.overview_tables:
            spec["class"] = "overview_table"
            config.results.append(spec)
        del config["overview_tables"]

    # results: default values
    for spec in config.results:
        spec.setdefault("class", "results_table")

        # set common default values
        spec.setdefault("show", True)
        spec.setdefault("clearpage", spec["class"] == "section")
        spec.setdefault("filter_regex", None)

        if spec["class"] == "section":
            spec.setdefault("name", "Section")
            spec.setdefault("pagewidth", None)
        elif spec["class"] == "results_table":
            spec.setdefault("metrics_legend", True)
            spec.setdefault("escape_latex_header", True)
            spec.setdefault("rotate_header", True)
            spec.setdefault("vertical_bars", True)
            spec.setdefault("export_latex", None)
            spec.setdefault("color_failed", "red")
            spec.setdefault("multirow", True)
            spec.setdefault("override_as_failed", [])
        elif spec["class"] == "summarize_sequences_table":
            spec.setdefault("header", "")
            spec.setdefault("export_latex", None)
            spec.setdefault("escape_latex_header", True)
            spec.setdefault("rotate_header", True)
        elif spec["class"] == "plot":
            spec.setdefault("plot_ate", False)
            spec.setdefault("figsize", None)
            spec.setdefault("title", None)
            spec.setdefault("reference_experiment", None)
            spec.setdefault("width", None)
            spec.setdefault("ylim", Munch())
            spec.ylim.setdefault("top", None)
            spec.ylim.setdefault("bottom", None)
            spec.setdefault("ylim_cost", Munch())
            spec.ylim_cost.setdefault("top", None)
            spec.ylim_cost.setdefault("bottom", None)
            spec.setdefault("ylim_ate", Munch())
            spec.ylim_ate.setdefault("top", None)
            spec.ylim_ate.setdefault("bottom", None)
            spec.setdefault("ylim_tolerance", Munch())
            spec.ylim_tolerance.setdefault("top", None)
            spec.ylim_tolerance.setdefault("bottom", None)
            spec.setdefault("xlim_time", Munch())
            spec.xlim_time.setdefault("right", None)
            spec.xlim_time.setdefault("left", None)
            spec.setdefault("xlim_time_fastest", Munch())
            spec.xlim_time_fastest.setdefault("right", None)
            spec.xlim_time_fastest.setdefault("left", None)
            spec.setdefault("xlim_it", Munch())
            spec.xlim_it.setdefault("right", None)
            spec.xlim_it.setdefault("left", None)
            spec.setdefault("xlimits", Munch())
            spec.xlimits.setdefault("right", None)
            spec.xlimits.setdefault("left", None)
            spec.setdefault("legend_loc", "best")
            spec.setdefault("align_fraction", None)
            spec.setdefault("layout", "horizontal")
            spec.setdefault("extend_x", False)
            if "problem_size_variants" not in spec and "memory_variants" in spec:
                # legacy support for "memory_variants"
                spec.problem_size_variants = spec.memory_variants
                del spec["memory_variants"]
            spec.setdefault("problem_size_variants", ["cam", "lm", "obs"])
            spec.setdefault("bal_cost_include", ["cost_time", "cost_it", "tr_radius", "inner_it", "memory"])
            spec.setdefault("tolerances", [0.01, 0.001, 0.00001])
            spec.setdefault("plot_tolerances", False)
            spec.setdefault("best_fit_line", True)
            spec.setdefault("reverse_zorder", False)
            spec.setdefault("plot_cost_semilogy", True)
            spec.setdefault("marker_size", 8)
            spec.setdefault("ylabel", True)
            spec.setdefault("suptitle", None)
            spec.setdefault("rotate2d", 0)
            spec.setdefault("trajectory_axes", "xy")
        elif spec["class"] == "overview_table":
            spec.setdefault("export_latex", None)

    # expand templates in path names
    template_args = dict(config_dir=os.path.dirname(os.path.abspath(path)))
    for key in ["base_path", "output_path", "cache_dir"]:
        config.options[key] = Template(config.options[key]).substitute(**template_args)
    if isinstance(config.options.import_experiments, str):
        config.options.import_experiments = [config.options.import_experiments]
    config.options.import_experiments = [
        Template(path).substitute(**template_args) for path in config.options.import_experiments
    ]

    # import experiments
    imported_experiments = []
    for path in config.options.import_experiments:
        cfg = load_experiments_config(path, args)
        imported_experiments.extend(cfg.experiments)
    config.experiments = imported_experiments + config.experiments

    return config
