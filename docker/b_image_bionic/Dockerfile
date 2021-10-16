FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget gnupg lsb-release software-properties-common && apt-get clean all

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add -
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-11 main" > /etc/apt/sources.list.d/llvm11.list
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-12 main" > /etc/apt/sources.list.d/llvm12.list

RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake git libtbb-dev libeigen3-dev libglew-dev ccache libjpeg-dev libpng-dev openssh-client liblz4-dev libbz2-dev libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev libopencv-dev libpython2.7-dev libgtest-dev lsb-core gcovr ggcov lcov librealsense2-dev librealsense2-gl-dev librealsense2-dkms librealsense2-utils doxygen graphviz libsuitesparse-dev clang-11 clang-format-11 clang-tidy-11 clang-12 clang-format-12 clang-tidy-12 g++-9 libfmt-dev && apt-get clean all

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-9
RUN gcc --version
RUN g++ --version