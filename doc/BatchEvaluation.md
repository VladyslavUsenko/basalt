# Batch Evaluation of Square Root Optimization and Marginalization

In this tutorial we detail how you can use the batch evaluation
scripts to reproduce the results of the ICCV'21 paper Demmel et al.,
"Square Root Marginalization for Sliding-Window Bundle Adjustment".
In the paper we discuss how square root estimation techniques can be
used in Basalt's optimization-based sliding-window odometry to make
optimization faster and marginalization numerially more stable.  See
the [project page](https://go.vision.in.tum.de/rootvo) for further
details.

Basalt's VIO/VO now runs with single-precision floating point numbers
by default, using the new square root formulation. The conventional
squared (Hessian-based) formualtion is still available via config
options. For manual testing, you can pass `--use-double true` or
`--use-double false` (default) as command line arguments to
`basalt_vio`, and change `config.vio_sqrt_marg` in the config file,
which controls if the marginalization prior is stored in Hessian or
Jacobian form (default: `true`), as well as
`config.vio_linearization_type` in the config file, which controls
whether to use Schur complement, or nullspace projection and
QR-decomposition for optimization and marginalization (`"ABS_SC"` or
`"ABS_QR"` (default)).

In the following tutorial we systematically compare the different
formulations in single and double precision to reproduce the results
from the ICCV'21 paper. You can of course adjust the correspondig
config files to evaluate other aspects of the system.

## Prerequisites

  1. **Source installation of Basalt:** The batch evaluation scripts
     by default assume that the `build` folder is directly inside the
     source checkout `basalt`. See
     [README.md](../README.md#source-installation-for-ubuntu-1804-and-macos-1014-mojave)
     for instructions.
  2. **Downloads of the datasets:** We evaluate EuRoC (all 11
     sequneces), TUMVI (euroc format in 512x512 resultion; sequences:
     corridor1-2, magistrale1-2, room1-2, slides1-2), and Kitti
     Odometry (sequences 00-10).  It's recommended to store the data
     locally on an SSD to ensure that reading the images is not the
     bottleneck during evaluation (on a multicore desktop Basalt runs
     many times faster than real-time). There are instructions for
     downloading these dataset: [EuRoC](VioMapping.md#euroc-dataset),
     [TUMVI](VioMapping.md#tum-vi-dataset),
     [KITTI](Vo.md#kitti-dataset). Calibration for EuRoC and TUMVI is
     provided in the `data` folder. For KITTI you can use the
     `basalt_convert_kitti_calib.py` script to convert the provided
     calibration to a Basalt-compatible format (see
     [KITTI](Vo.md#kitti-dataset)).
  3. **Dependencies of evaluation scripts:** You need pip packages
     `py_ubjson`, `matplotlib`, `numpy`, `munch`, `scipy`, `pylatex`,
     `toml`. How to install depends on your Python setup (virtualenv,
     conda, ...). To just install for the local user with pip you can
     use the command `python3 -m pip install --user -U py_ubjson
     matplotlib numpy munch scipy pylatex toml`. For generating result
     tables and plots you additionally need latexmk and a LaTeX
     distribution (Ubuntu: `sudo apt install texlive-latex-extra
     latexmk`; macOS with Homebrew: `brew install --cask mactex`).

## Folder Structure

The batch evaluation scripts and config files assume a certain folder
structure inside a "parent" folder, since relative paths are used to
find the compiled executable and calibration files. So **it's
important to follow the folder structure**.

```
parent-folder/
├─ basalt/
│  ├─ build/
│  │  ├─ basalt_vio
│  │  ├─ ...
│  ├─ data/
│  │  ├─ euroc_ds_calib.json
│  │  ├─ ...
│  ├─ ...
├─ experiments/
│  ├─ iccv_tutorial/
│  │  ├─ basalt_batch_config.toml
│  │  ├─ experiments-iccv.toml
│  │  ├─ 01_iccv_all/
│  │  │  ├─ ...
│  │  ├─ 02_iccv_runtime/
│  │  │  ├─ ...
```

As a sibling of the `basalt` source checkout we'll have an
`experiments` folder, and inside, a folder `iccv_tutorial` for this
tutorial. Into that folder, we copy the provided
`basalt_batch_config.toml` file that defines the configurations we
want to evaluate and from which we generate individual config files
for each VIO / VO run. We also copy the provided
`experiments-iccv.toml` config file, which defines the results tables
and plots that we generate from the experiments' logs.

> *Note:* Commands in this tutorial are assumed to be executed from
> within `parent-folder` unless specified otherwise.

```bash
mkdir -p experiments/iccv_tutorial
cp basalt/data/iccv21/basalt_batch_config.toml experiments/iccv_tutorial/
cp basalt/data/iccv21/experiments-iccv.toml experiments/iccv_tutorial/
```

## Generate Experimental Configs

First, edit the copied configuration file
`experiments/iccv_tutorial/basalt_batch_config.toml` and modify all
`"dataset-path"` lines to point to the locations where you downloaded
the datasets to.

Now, we can generate per-experiment config files:

```bash
cd experiments/iccv_tutorial/
../../basalt/scripts/batch/generate-batch-configs.py .
```

This will create subfolder `01_iccv_all` containing folders
`vio_euroc`, `vio_tumvi`, and `vo_kitti`, which in turn contain all
generated `basalt_config_...json` files, one for each experiment we
will run.

## Run Experiments

We can now run all experiments for those generate configs. Each config
/ sequence combination will automatically be run twice and only the
second run is evaluated, which is meant to ensure that file system
caches are hot.

Since we also evaluate runtimes, we recommend that you don't use the
machine running the experiments for anything else and also ensure no
expensive background tasks are running during the evaluation. On one
of our desktops with 12 (virtual) cores the total evaluation of all
sequences takes aroudn 3:30h. Your milage may vary of course depending
on the hardware.

```bash
cd experiments/iccv_tutorial/
time ../../basalt/scripts/batch/run-all-in.sh 01_iccv_all/
```

Inside `01_iccv_all`, a new folder with the start-time of the
experimental run is created, e.g., `20211006-143137`, and inside that
you can again see the same per-dataset subfolders `vio_euroc`,
`vio_tumvi`, and `vo_kitti`, inside of which there is a folder for
each config / run. Inside these per-run folders you can find log files
including the command line output, which you can inspect in case
something doesn't work.

In a second terminal, you can check the status of evaluation while it
is running (adjust the argument to match the actual folder name).

```bash
cd experiments/iccv_tutorial/
../../basalt/scripts/batch/list-jobs.sh 01_iccv_all/20211006-143137
```

If you see failed experiments for the square root solver in single
precision, don't worry, that is expected.

## Generate Results Tables and Plots

After all experimental runs have completed, you can generate a PDF
file with tabulated results and plots, similar to those in the ICCV'21
paper.

```bash
cd experiments/iccv_tutorial/
../../basalt/scripts/batch/generate-tables.py --config experiments-iccv.toml --open
```

The results are in the generated `tables/experiments-iccv.pdf` file
(and with the `--open` argument should automatically open with the
default PDF reader).

## Better Runtime Evaluation

The experiments above have the extended logging of eigenvalue and
nullspace information enabled, which does cost a little extra
runtime. To get a better runtime comparison, you can re-run the
experiments without this extended logging. The downside is, that you
can only generate the results tables, but not the plots.

We assume that you have already followed the tutorial above, including
the initial folder setup. For these modified experiments, we redo all
three steps (generating config files; running experiments; generating
results) with slight modifications.

First, edit the `experiments/iccv_tutorial/basalt_batch_config.toml`
file at the bottom, and uncomment the commented entries in
`_batch.combinations` as well as the commented `revision`. At the same
time, comment out the initially uncommented lines. It should look
something like this after the modifications:

```toml
[_batch.combinations]
#vio_euroc = ["vio",             "savetumgt", "extlog", "runtwice", "all_meth", "all_double", "all_euroc"]
#vio_tumvi = ["vio", "tumvivio", "savetumgt", "extlog", "runtwice", "all_meth", "all_double", "more_tumvi"]
#vo_kitti  = ["vo",  "kittivo",  "savetumgt", "extlog", "runtwice", "all_meth", "all_double", "all_kitti"]

vio_euroc = ["vio",             "runtwice", "all_meth", "all_double", "all_euroc"]
vio_tumvi = ["vio", "tumvivio", "runtwice", "all_meth", "all_double", "more_tumvi"]
vo_kitti  = ["vo",  "kittivo",  "runtwice", "all_meth", "all_double", "all_kitti"]
```

```toml
[_batch]
#revision = "01_iccv_all"
revision = "02_iccv_runtime"
```

You can see that we removed the `savetumgt` and `extlog` named config
elements and that generated config files and results for this second
run of experiments will be placed in `02_iccv_runtime`.

Now generate config files and start the experimental runs:

```
cd experiments/iccv_tutorial/
../../basalt/scripts/batch/generate-batch-configs.py .
time ../../basalt/scripts/batch/run-all-in.sh 02_iccv_runtime/
```

Before generating the results PDF you need to now edit the
`experiments-iccv.toml` file, point it to the new location for
experimental logs and disable the generation of plots. Check the place
towards the start of the file where substitutions for
`EXP_PATTERN_VIO` and `EXP_PATTERN_VO` are defined, as well as
`SHOW_TRAJECTORY_PLOTS`, `SHOW_EIGENVALUE_PLOTS`, and
`SHOW_NULLSPACE_PLOTS`. After your modifications, that section should
look something like:

```toml
###################
## where to find experimental runs
[[substitutions]]

#EXP_PATTERN_VIO = "01_iccv_all/*-*/vio_*/"
#EXP_PATTERN_VO = "01_iccv_all/*-*/vo_*/"

EXP_PATTERN_VIO = "02_iccv_runtime/*-*/vio_*/"
EXP_PATTERN_VO = "02_iccv_runtime/*-*/vo_*/"


###################
## which kind of plots to show
[[substitutions]]
SHOW_TRAJECTORY_PLOTS = false
SHOW_EIGENVALUE_PLOTS = false
SHOW_NULLSPACE_PLOTS = false
```

Now we can generate the results tables for the new experimental runs
with the same command as before:

```bash
cd experiments/iccv_tutorial/
../../basalt/scripts/batch/generate-tables.py --config experiments-iccv.toml --open
```
