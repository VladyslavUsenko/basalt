FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget gnupg lsb-release software-properties-common && apt-get clean all

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add -
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-14 main" > /etc/apt/sources.list.d/llvm11.list
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-15 main" > /etc/apt/sources.list.d/llvm12.list

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake git libtbb-dev libeigen3-dev libglew-dev ccache libjpeg-dev libpng-dev openssh-client liblz4-dev libbz2-dev libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev libopencv-dev libpython2.7-dev libgtest-dev lsb-core gcovr lcov librealsense2-dev librealsense2-gl-dev librealsense2-dkms librealsense2-utils doxygen graphviz libsuitesparse-dev clang-14 clang-format-14 clang-tidy-14 clang-15 clang-format-15 clang-tidy-15 libfmt-dev libstdc++-12-dev && apt-get clean all
