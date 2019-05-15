

### Clang-format
We use clang-format to maintain a consistent formating of the code. Since there are small differences between different version of clang-format we use version 8 on all platforms.

On Ubuntu 18.04 run the following commands to install clang-format-8
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
sudo sh -c 'echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-8 main" > /etc/apt/sources.list.d/llvm8.list'
sudo apt-get install clang-format-8
```

On MacOS [Homebrew](https://brew.sh/) should install the right version of clang-format:
```
brew install clang-format
```

### Install and configure QtCreator
Download and install QtCreator. On Ubuntu 18.04 run:
```
wget https://download.qt.io/official_releases/qtcreator/4.9/4.9.0/qt-creator-opensource-linux-x86_64-4.9.0.run
chmod +x qt-creator-opensource-linux-x86_64-4.9.0.run
./qt-creator-opensource-linux-x86_64-4.9.0.run
```

On MacOS run:
```
wget https://download.qt.io/official_releases/qtcreator/4.9/4.9.0/qt-creator-opensource-mac-x86_64-4.9.0.dmg
open qt-creator-opensource-mac-x86_64-4.9.0.dmg
```

After installation, go to `Help` -> `About plugins...` in the menu and enable Beautifier plugin (formats the code automatically on save):

![qt_creator_plugins](/doc/img/qt_creator_plugins.png)

Go to `Tools` -> `Options` and select the Beautifier tab. There select ClangFormat as the tool in `General` tab.

![qt_creator_beautifier_general](/doc/img/qt_creator_beautifier_general.png)

Select file as predefined style in `Clang Format` tab. Also select `None` as the fallback style. **For Ubuntu 18.04** change the executable name to `/usr/bin/clang-format-8`.

![qt_creator_beautifier_clang_format](/doc/img/qt_creator_beautifier_clang_format.png)

### Build project
First, clone the project repository.
```
git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git
```

After that, in QtCreator open to the `CMakeLists.txt` in the `basalt` folder and configure the project with `Release with Debug Info` configuration. The build directory should point to `/<your_installation_path>/basalt/build`.

![qt_creator_configure_project](/doc/img/qt_creator_configure_project.png)

Finally, you should be able to build and run the project.

