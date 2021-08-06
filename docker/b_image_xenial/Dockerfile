FROM ubuntu:16.04

RUN apt-get update && apt-get install -y cmake git libtbb-dev libeigen3-dev libglew-dev ccache libjpeg-dev libpng12-dev openssh-client liblz4-dev libbz2-dev libboost-regex-dev libboost-filesystem-dev libboost-date-time-dev libboost-program-options-dev libopencv-dev libpython2.7-dev wget libgtest-dev lsb-release software-properties-common && apt-get clean all

RUN add-apt-repository ppa:ubuntu-toolchain-r/test

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
RUN add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u


RUN apt update && apt-get install -y g++-7 librealsense2-dev librealsense2-gl-dev librealsense2-dkms librealsense2-utils && apt-get clean all


RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7
RUN gcc --version
RUN g++ --version

RUN wget https://cmake.org/files/v3.13/cmake-3.13.4-Linux-x86_64.sh
RUN chmod +x cmake-3.13.4-Linux-x86_64.sh
RUN ./cmake-3.13.4-Linux-x86_64.sh  --skip-license --prefix=/usr/local
RUN update-alternatives --install /usr/bin/cmake cmake /usr/local/bin/cmake 1 --force
RUN cmake --version