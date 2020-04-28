FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget gnupg 

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add -
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-10 main" > /etc/apt/sources.list.d/llvm10.list

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE
RUN echo "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" > /etc/apt/sources.list.d/realsense.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake git libtbb-dev libeigen3-dev libglew-dev ccache libjpeg-dev libpng-dev openssh-client liblz4-dev libbz2-dev libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev libopencv-dev libpython2.7-dev libgtest-dev lsb-core gcovr ggcov lcov librealsense2-dev librealsense2-gl-dev librealsense2-dkms librealsense2-utils doxygen graphviz libsuitesparse-dev clang-10 clang-format-10
