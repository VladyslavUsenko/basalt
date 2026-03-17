FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG LLVM_VERSION=21
ARG RUST_VERSION=1.93.1

ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV VCPKG_DEFAULT_TRIPLET=x64-linux
ENV PATH="/root/.cargo/bin:/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"
ENV CC="clang-${LLVM_VERSION}"
ENV CXX="clang++-${LLVM_VERSION}"

SHELL ["/bin/bash", "-lc"]

RUN set -euxo pipefail \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      curl \
      wget \
      zip \
      unzip \
      tar \
      pkg-config \
      build-essential \
      ccache \
      python3-pip \
      gnupg \
      lsb-release \
      doxygen \
      graphviz \
      autoconf \
      autoconf-archive \
      automake \
      bison \
      libtool \
      libltdl-dev \
      ninja-build \
      openssh-client \
      python3-venv \
      libgl1-mesa-dev \
      libglu1-mesa-dev \
      libx11-dev \
      libxcursor-dev \
      libxinerama-dev \
      libxrandr-dev \
      libxi-dev \
      libxtst-dev \
      libftdi-dev \
      libusb-1.0-0-dev \
      libudev-dev \
    && curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc >/dev/null \
    && echo "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      "clang-${LLVM_VERSION}" \
      "libclang-rt-${LLVM_VERSION}-dev" \
      "lld-${LLVM_VERSION}" \
      "llvm-${LLVM_VERSION}" \
      "llvm-${LLVM_VERSION}-tools" \
      "clang-format-${LLVM_VERSION}" \
      "clang-tidy-${LLVM_VERSION}" \
    && update-alternatives --install /usr/bin/cc cc "/usr/bin/clang-${LLVM_VERSION}" 100 \
    && update-alternatives --install /usr/bin/c++ c++ "/usr/bin/clang++-${LLVM_VERSION}" 100 \
    && update-alternatives --set cc "/usr/bin/clang-${LLVM_VERSION}" \
    && update-alternatives --set c++ "/usr/bin/clang++-${LLVM_VERSION}" \
    && python3 -m pip install --no-cache-dir cmake pre-commit \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}" \
    && rustup component add rustfmt clippy \
    && command -v "clang-${LLVM_VERSION}" \
    && command -v cc \
    && command -v c++ \
    && command -v llvm-profdata \
    && command -v cargo \
    && rustc --version \
    && cargo --version \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
