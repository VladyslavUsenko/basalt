[![pipeline status](https://gitlab.com/VladyslavUsenko/basalt/badges/master/pipeline.svg)](https://gitlab.com/VladyslavUsenko/basalt/commits/master)

## Basalt
For more information see https://vision.in.tum.de/research/vslam/basalt

![teaser](doc/img/teaser.png)

This project contains tools for:
* Camera, IMU and motion capture calibration.
* Visual-inertial odometry and mapping.
* Simulated environment to test different components of the system.

Some reusable components of the system are available as a separate [header-only library](https://gitlab.com/VladyslavUsenko/basalt-headers) ([Documentation](https://vladyslavusenko.gitlab.io/basalt-headers/)).

There is also a [Github mirror](https://github.com/VladyslavUsenko/basalt-mirror) of this project to enable easy forking.

## Related Publications
Visual-Inertial Odometry and Mapping:
* **Visual-Inertial Mapping with Non-Linear Factor Recovery**, V. Usenko, N. Demmel, D. Schubert, J. Stückler, D. Cremers, In IEEE Robotics and Automation Letters (RA-L) [[DOI:10.1109/LRA.2019.2961227]](https://doi.org/10.1109/LRA.2019.2961227) [[arXiv:1904.06504]](https://arxiv.org/abs/1904.06504).

Calibration (explains implemented camera models):
* **The Double Sphere Camera Model**, V. Usenko and N. Demmel and D. Cremers, In 2018 International Conference on 3D Vision (3DV), [[DOI:10.1109/3DV.2018.00069]](https://doi.org/10.1109/3DV.2018.00069), [[arXiv:1807.08957]](https://arxiv.org/abs/1807.08957).

Calibration (demonstrates how these tools can be used for dataset calibration):
* **The TUM VI Benchmark for Evaluating Visual-Inertial Odometry**, D. Schubert, T. Goll,  N. Demmel, V. Usenko, J. Stückler, D. Cremers, In 2018 International Conference on Intelligent Robots and Systems (IROS), [[DOI:10.1109/IROS.2018.8593419]](https://doi.org/10.1109/IROS.2018.8593419), [[arXiv:1804.06120]](https://arxiv.org/abs/1804.06120).

Calibration (describes B-spline trajectory representation used in camera-IMU calibration):
* **Efficient Derivative Computation for Cumulative B-Splines on Lie Groups**, C. Sommer, V. Usenko, D. Schubert, N. Demmel, D. Cremers, In 2020 Conference on Computer Vision and Pattern Recognition (CVPR), [[DOI:10.1109/CVPR42600.2020.01116]](https://doi.org/10.1109/CVPR42600.2020.01116), [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860).

Optimization (describes square-root optimization and marginalization used in VIO/VO):
* **Square Root Marginalization for Sliding-Window Bundle Adjustment**, N. Demmel, D. Schubert, C. Sommer, D. Cremers, V. Usenko, In 2021 International Conference on Computer Vision (ICCV), [[arXiv:2109.02182]](https://arxiv.org/abs/2109.02182)


## Installation
### Binary installation from GitLab releases (Ubuntu 22.04+ amd64, MacOS 26+ arm64)
Install the latest published release into `~/.local`:
```
curl -LsSf https://gitlab.com/VladyslavUsenko/basalt/-/raw/master/scripts/install.sh | sh
```
The installer places binaries in `~/.local/bin`, libraries in `~/.local/lib`, and data files in `~/.local/etc/basalt`.

### Source installation (CMake presets + vcpkg)
Clone the source code with the `thirdparty/vcpkg` submodule, then build with CMake presets. Install CMake (>= 3.24), Ninja, and a C++ compiler first.
```
git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git
cd basalt
# If you cloned without --recursive, fetch the vcpkg submodule:
# git submodule update --init thirdparty/vcpkg

# Bootstrap vcpkg once if needed:
# ./thirdparty/vcpkg/bootstrap-vcpkg.sh -disableMetrics

cmake --preset relwithdebinfo
cmake --build --preset relwithdebinfo -j8
ctest --preset relwithdebinfo
```

By default presets use:
`thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake`

On macOS, build the release preset and then package the release artifact:
```
cmake --preset release
cmake --build --preset release -j8
./scripts/package_macos_release.sh <tag>
# Optional: upload the generated tarball and checksums to the GitLab release for <tag>
./scripts/package_macos_release.sh <tag>
```

## Usage
* [Camera, IMU and Mocap calibration. (TUM-VI, Euroc, UZH-FPV and Kalibr datasets)](doc/Calibration.md)
* [Visual-inertial odometry and mapping. (TUM-VI and Euroc datasets)](doc/VioMapping.md)
* [Visual odometry (no IMU). (KITTI dataset)](doc/Vo.md)
* [Simulation tools to test different components of the system.](doc/Simulation.md)
* [Batch evaluation tutorial (ICCV'21 experiments)](doc/BatchEvaluation.md)

## Device support
* [Tutorial on Camera-IMU and Motion capture calibration with Realsense T265.](doc/Realsense.md)

## Development
* [Development environment setup.](doc/DevSetup.md)

## Licence
The code is provided under a BSD 3-clause license. See the LICENSE file for details.
Note also the different licenses of thirdparty code.

Some improvements are ported back from the fork
[granite](https://github.com/DLR-RM/granite) (MIT license).
