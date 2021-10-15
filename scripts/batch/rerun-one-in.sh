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

set -e
set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

FOLDER="${1}"

cd "$FOLDER"

# backup previous files
DATE=`date +'%Y%m%d-%H%M%S'`
BACKUP_FOLDER=results-backup-$DATE
for f in *.jobid *.log stats*.*json; do
    if [ -f $f ]; then
        mkdir -p $BACKUP_FOLDER
        mv $f $BACKUP_FOLDER/
    fi
done

echo "Created" > status.log
echo "Restarted" >> status.log

echo "Starting run in $PWD"
"$SCRIPT_DIR"/run-one.sh "$PWD"
