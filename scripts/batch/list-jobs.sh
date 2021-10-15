#!/usr/bin/env bash
##
## BSD 3-Clause License
##
## This file is part of the Basalt project.
## https://gitlab.com/VladyslavUsenko/basalt.git
##
## Copyright (c) 2019-2021, Vladyslav Usenko and Nikolaus Demmel.
## All rights reserved.
##

#
# Usage:
#    list-jobs.sh DIRNAME [DIRNAME ...] [-s|--short] [-o|--only STATUS]
#
# Lists all batch jobs found in DIRNAME. If the optional argument
# STATUS is passed, only lists jobs with that status. Multiple
# statuses can be passed in a space-separated string.
#
# Possible status arguments: queued, running, completed, failed, unknown
# You can also use 'active' as a synonym for 'queued running unknown'

# exit on error
set -o errexit -o pipefail


# we need GNU getopt...
GETOPT=getopt
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [ -f /usr/local/opt/gnu-getopt/bin/getopt ]; then
        GETOPT="/usr/local/opt/gnu-getopt/bin/getopt"
    fi
fi

# option parsing, see: https://stackoverflow.com/a/29754866/1813258
usage() { echo "Usage: `basename $0` DIRNAME [DIRNAME ...] [-s|--short] [-o|--only STATUS]" ; exit 1; }

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! "$GETOPT" --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

OPTIONS=hsjo:
LONGOPTS=help,short,jobids,only:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$("$GETOPT" --options=$OPTIONS --longoptions=$LONGOPTS --name "`basename $0`" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    usage
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

SHORT=n
ONLY=""
JOBIDS=n
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -h|--help) usage ;;
        -s|--short) SHORT=y; shift ;;
        -j|--jobids) JOBIDS=y; shift ;;
        -o|--only) ONLY="$2"; shift 2 ;;
        --) shift; break ;;
        *) echo "Programming error"; exit 3 ;;
    esac
done

# handle non-option arguments --> directories
if [[ $# -lt 1 ]]; then
    echo "Error: Pass at least one folder"
    usage
fi
DIRS=("$@")

# status aliases:
ONLY="${ONLY/active/queued running unknown}"
ONLY="${ONLY/notcompleted/queued running failed unknown}"

contains() {
    [[ $1 =~ (^| )$2($| ) ]] && return 0 || return 1
}

display() {
    if [ -z "$ONLY" ] || contains "$ONLY" $2; then
        if [ $SHORT = y ]; then
            echo "$1"
        else
            echo -n "$1 : $2"
            if [ -n "$3" ]; then
                echo -n " - $3"
            fi
            echo ""
        fi
    fi
}

for d in "${DIRS[@]}"; do
    for f in `find "$d" -name status.log | sort`; do
        DIR=`dirname "$f"`

        # ignore backup folder from "rerun" scripts
        if [[ `basename $DIR` = results-backup* ]]; then
            continue
        fi

        if ! grep Started "$f" > /dev/null; then
            display "$DIR" unknown "not started"
            continue
        fi

        # job has started:

        if grep Completed "$f" > /dev/null ; then
            display "$DIR" completed ""
            continue
        fi

        # job has started, but not completed (cleanly)

        # check signs of termination
        if [ -f "$DIR"/output.log ] && grep "Command terminated by signal" "$DIR"/output.log > /dev/null; then
            display "$DIR" failed killed "`grep -oP 'Command terminated by \Ksignal .+' "$DIR"/output.log`"
            continue
        fi

        # might be running or aborted
        display "$DIR" unknown started

    done
done
