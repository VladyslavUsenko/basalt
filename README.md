## Basalt
For more information see https://vision.in.tum.de/research/vslam/basalt

![teaser](doc/img/teaser.png)

This project contains tools for:
* Camera, IMU and motion capture calibration.
* Visual-inertial odometry and mapping.
* Simulated environment to test different components of the system.


## Related Publications
Visual-Inertial Odometry and Mapping:
* **Visual-Inertial Mapping with Non-Linear Factor Recovery**, V. Usenko, N. Demmel, D. Schubert, J. Stückler, D. Cremers, In [[arXiv:]](https://arxiv.org/abs/).

Calibration (explains implemented camera models):
* **The Double Sphere Camera Model**, V. Usenko and N. Demmel and D. Cremers, In 2018 International Conference on 3D Vision (3DV), [[DOI:10.1109/3DV.2018.00069]](https://doi.org/10.1109/3DV.2018.00069), [[arXiv:1807.08957]](https://arxiv.org/abs/1807.08957).

Calibration (demonstrates how these tools can be used for dataset calibration):
* **The TUM VI Benchmark for Evaluating Visual-Inertial Odometry**, D. Schubert, T. Goll,  N. Demmel, V. Usenko, J. Stückler, D. Cremers, In 2018 International Conference on Intelligent Robots and Systems (IROS), [[DOI:10.1109/IROS.2018.8593419]](https://doi.org/10.1109/IROS.2018.8593419), [[arXiv:1804.06120]](https://arxiv.org/abs/1804.06120).


## Installation
### APT installation for Ubuntu 16.04 and 18.04 (Fast)
Set up keys
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0D97B6C9
```
Add the repository to the sources list. On **Ubuntu 18.04** run:
```
sudo sh -c 'echo "deb [arch=amd64] http://packages.usenko.eu/ubuntu bionic main" > /etc/apt/sources.list.d/basalt.list'
```
On **Ubuntu 16.04** run:
```
sudo sh -c 'echo "deb [arch=amd64] http://packages.usenko.eu/ubuntu xenial main" > /etc/apt/sources.list.d/basalt.list'
```
Update the Ubuntu package index and install Basalt:
```
sudo apt-get update
sudo apt-get install basalt
```
### Source installation for Ubuntu 18.04 and MacOS 10.14 Mojave
Clone the source code for the project
```
git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git
cd basalt
./scripts/install_deps.sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```
NOTE: It is possible to compile the code on Ubuntu 16.04, but you need to install cmake-3.10 and gcc-7. See corresponding [Dockerfile](docker/b_image_xenial/Dockerfile) as an example.

## Usage
* [Camera, IMU and Mocap calibration.](doc/Calibration.md)
* [Visual-inertial odometry and mapping.](doc/VioMapping.md)
* [Simulation tools to test different components of the system.](doc/Simulation.md)

## Licence
The code for this practical course is provided under a BSD 3-clause license. See the LICENSE file for details.
Note also the different licenses of thirdparty submodules.
