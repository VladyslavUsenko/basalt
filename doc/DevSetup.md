

### Clang-format
We use clang-format to maintain a consistent formating of the code. Since there are small differences between different version of clang-format we use version 11 on all platforms.

On **Ubuntu 20.04 or 18.04** run the following commands to install clang-format-11
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo sh -c 'echo "deb http://apt.llvm.org/$(lsb_release -sc)/ llvm-toolchain-$(lsb_release -sc)-11 main" > /etc/apt/sources.list.d/llvm11.list'
sudo apt-get update
sudo apt-get install clang-format-11
```

On **MacOS** [Homebrew](https://brew.sh/) should install the right version of clang-format:
```
brew install clang-format
```

### Realsense Drivers (Optional)
If you want to use the code with Realsense T265 cameras you should install the realsense library.

On **Ubuntu 20.04 or 18.04** run the following commands:
```
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE
sudo sh -c 'echo "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo bionic main" > /etc/apt/sources.list.d/realsense.list'
sudo apt-get update
sudo apt-get install librealsense2-dev librealsense2-gl-dev librealsense2-dkms librealsense2-utils librealsense2-dkms
```

On **MacOS** run:
```
brew install librealsense
```

### Install and configure QtCreator
Download and install QtCreator. On **Ubuntu 20.04 or 18.04** run:
```
wget https://download.qt.io/official_releases/qtcreator/4.10/4.10.0/qt-creator-opensource-linux-x86_64-4.10.0.run
chmod +x qt-creator-opensource-linux-x86_64-4.10.0.run
./qt-creator-opensource-linux-x86_64-4.10.0.run
```

On **MacOS** run:
```
brew cask install qt-creator
```

After installation, go to `Help` -> `About plugins...` in the menu and enable Beautifier plugin (formats the code automatically on save):

![qt_creator_plugins](/doc/img/qt_creator_plugins.png)

Go to `Tools` -> `Options` and select the Beautifier tab. There select ClangFormat as the tool in `General` tab.

![qt_creator_beautifier_general](/doc/img/qt_creator_beautifier_general.png)

Select file as predefined style in `Clang Format` tab. Also select `None` as the fallback style. For **Ubuntu 20.04 or 18.04** change the executable name to `/usr/bin/clang-format-11`.

![qt_creator_beautifier_clang_format](/doc/img/qt_creator_beautifier_clang_format.png)

### Build project
First, clone the project repository.
```
git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git
```

After that, in QtCreator open to the `CMakeLists.txt` in the `basalt` folder and configure the project with `Release with Debug Info` configuration. The build directory should point to `/<your_installation_path>/basalt/build`.

![qt_creator_configure_project](/doc/img/qt_creator_configure_project.png)

Finally, you should be able to build and run the project.

