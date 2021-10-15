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
# Example usage:
#     $ ./query-config.py path/to/basalt_config.json value0.\"config.vio_debug\"
#     10G

import json
import toml
import argparse
import sys


def parse_query(query):
    query_list = []

    quote_open_char = None
    curr = ""
    for c in query:
        if quote_open_char:
            if c == quote_open_char:
                quote_open_char = None
            else:
                curr += c
        elif c in ['"', "'"]:
            quote_open_char = c
        elif c == '.':
            query_list.append(curr)
            curr = ""
        else:
            curr += c
    query_list.append(curr)

    return query_list


def query_config(path, query, default_value=None, format_env=False, format_cli=False):
    query_list = parse_query(query)
    with open(path) as f:
        cfg = json.load(f)
    try:
        curr = cfg
        for q in query_list:
            curr = curr[q]
        result = curr
    except:
        if default_value is None:
            result = ""
        else:
            result = default_value
    if isinstance(result, dict):
        if format_env:
            lines = []
            for k, v in result.items():
                # NOTE: assumes no special escaping is necessary
                lines.append("{}='{}'".format(k, v))
            return "\n".join(lines)
        elif format_cli:
            args = ["--{} {}".format(k, v) for k, v in result.items()]
            return " ".join(args)
        else:
            result = toml.dumps(result)
    else:
        result = "{}".format(result)
    return result


def main():
    parser = argparse.ArgumentParser("Parse toml file and print content of query key.")
    parser.add_argument("config_path", help="path to toml file")
    parser.add_argument("query", help="query string")
    parser.add_argument("default_value", help="value printed if query is not successful", nargs='?')
    parser.add_argument(
        "--format-env",
        action="store_true",
        help="Expect dictionary as query result and output like environment variables, i.e. VAR='VALUE' lines.")
    parser.add_argument("--format-cli",
                        action="store_true",
                        help="Expect dictionary as query result and output like cli arguments, i.e. --VAR 'VALUE'.")
    args = parser.parse_args()

    res = query_config(args.config_path,
                       args.query,
                       default_value=args.default_value,
                       format_env=args.format_env,
                       format_cli=args.format_cli)
    print(res)


if __name__ == "__main__":
    main()
