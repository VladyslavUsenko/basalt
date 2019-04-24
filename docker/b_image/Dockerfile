FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget gnupg 

RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key| apt-key add -
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-8 main" > /etc/apt/sources.list.d/llvm8.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake git libtbb-dev libeigen3-dev libglew-dev ccache libjpeg-dev libpng-dev openssh-client liblz4-dev libbz2-dev libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev libopencv-dev libpython2.7-dev libgtest-dev lsb-core gcovr ggcov lcov clang-format-8
