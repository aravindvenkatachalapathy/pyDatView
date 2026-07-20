
## Installation

### Modern Qt/PyQtGraph install with Rust fast I/O
The current GUI uses `PySide6` and `pyqtgraph` for faster plotting and large overlays. Python 3.9 or newer is supported; the commands below use your `env314` environment with Python 3.13:

```bash
conda create -n env314 -c conda-forge python=3.13 pip numpy pandas polars pyarrow pyside6 pyqtgraph openpyxl xarray matplotlib chardet scipy
conda activate env314
python -m pip install -e .
```

For faster OpenFAST binary `.outb` and Bladed binary loading, install Rust and `maturin`, then build the optional Rust extension:

```bash
conda install -n env314 -c conda-forge rust maturin
conda activate env314
cd rust/pydatview_fastio
maturin develop --release
```

After this, the Python readers automatically use the Rust extension when `pydatview_fastio` is available, and fall back to the pure Python/NumPy readers otherwise.

### Windows installation
For Windows users, installer executables are available [here](https://github.com/ebranlard/pyDatView/releases) (look for the latest pyDatView\*.exe)

### Linux installation
The script is compatible with python 3 and relies on the following python packages: `numpy` `matplotlib`, `pandas`, `wxpython`.
To download the code and install the dependencies (with pip) run the following:
```bash
git clone https://github.com/ebranlard/pyDatView
cd pyDatView
python -m pip install --user -r requirements.txt
```
If the installation of `wxpython` fails, you may need to install the package python-wxgtk\* (e.g. `python-gtk3.0`) from your distribution. For Debian/Ubuntu systems, try:
`sudo apt-get install python-wxgtk3.0`.
For further troubleshooting you can check the [wxPython wiki page](https://wiki.wxpython.org/).

If the requirements are successfully installed you can run pyDatView by typing:
```bash
python pyDatView.py  # or pythonw pyDatView.py 
```
To easily access it later, you can add an alias to your `.bashrc` or install the pydatview module:
```bash
echo "alias pydat='python `pwd`/pyDatview.py'" >> ~/.bashrc
# or
python setup.py install
```


## MacOS installation
The installation should work with python3, with `brew` (with or without a `virtualenv`) or `anaconda`.
First, download the source code:
```bash
git clone https://github.com/ebranlard/pyDatView
cd pyDatView
```
Before installing the requirements, you need to be aware of the two following issues with MacOS:
- If you are using the native version of python, there is an incompatibility between the native version of `matplotlib` on MacOS and the version of `wxpython`. The solution is to use `virtualenv`, `brew` or `anaconda`.
- To use a GUI app, you need a python program that has access to the screen. These special python programs are in different locations. For the system-python, it's usually in `/System`, the `brew` versions are usually in `/usr/local/Cellar`, and the `anaconda` versions are usually called `python.app`.
The script `pythonmac` provided in this repository attempt to find the correct python program depending if you are in a virtual environment, in a conda environment, a system-python or a python from brew or conda. 

Different solutions are provided below depending on your preferred way of working.
For the latest Mac version, we recommend using anaconda.

### Anaconda-python version (outside a virtualenv)
The installation of anaconda sometimes replaces the system python with the anaconda version of python. You can see that by typing `which python`. Use the following:
```
python -m pip install --user -r requirements.txt # install requirements
conda install -c conda-forge wxpython            # install wxpython
pythonw pyDatView.py                             # NOTE: using pythonw not python
```
If the `pythonw` command above fails, try the few next options, and post an issue. You can try the `./pythonmac` provided in this repository
```bash
./pythonmac pyDatView.py
```
If that still doesn't work, you can try using the `python.app` from anaconda:
```bash
/anaconda3/bin/python.app
```
where `/anaconda3/bin/` is the path that would be returned by the command `which conda`. Note the `.app` at the end. If you don't have `python.app`, try installing it with `conda install -c anaconda python.app`


Note also that several users have been struggling to run pyDatView on the mac Terminal in new macOS systems. If you encounter the same issues, we recommend using the integrated zsh terminal from [VSCode](https://code.visualstudio.com) or using a more advanced terminal like [iterm2](https://iterm2.com/downloads.html) and perform the installation steps there. Also, make sure to stick to the base anaconda environment.





### Brew-python version (outside of a virtualenv)
If you have `brew` installed, and you installed python with `brew install python`, then the easiest is to use your `python3` version:
```
python3 -m pip install --user -r requirements.txt
python3 pyDatView.py
```

### Brew-python version (inside a virtualenv)
If you are inside a virtualenv, with python 3, use:
```
pip install -r requirements.txt
./pythonmac pyDatView.py
```
If the `pythonmac` commands fails, contact the developer, and in the meantime try to replace it with something like:
```
$(brew --prefix)/Cellar/python/XXXXX/Frameworks/python.framework/Versions/XXXX/bin/pythonXXX
```
where the result from `brew --prefix` is usually `/usr/loca/` and the `XXX` above corresponds to the version of python you are using in your virtual environment.



Note also that several users have been struggling to run pyDatView on the mac Terminal in new macOS systems. If you encounter the same issues, we recommend using the integrated zsh terminal from [VSCode](https://code.visualstudio.com) or using a more advanced terminal like [iterm2](https://iterm2.com/downloads.html) and perform the installation steps there. Also, make sure to stick to the base anaconda environment.

### Easy access
To easily access the program later, you can add an alias to your `.bashrc` or install the pydatview module:
```bash
echo "alias pydat='python `pwd`/pyDatview.py'" >> ~/.bashrc
# or
python setup.py install
```





## Adding more file formats
File formats can be added by implementing a subclass of `pydatview/io/File.py`, for instance `pydatview/io/VTKFile.py`. Existing examples are found in the folder `pydatview/io`.
Once implemented the fileformat needs to be registered in `pydatview/io/__init__.py` by adding an import line at the beginning of this script and adding a line in the function `fileFormats()` of the form `formats.append(FileFormat(VTKFile))`

If you believe your fileformat will be beneficial to the wind energy community, we recommend to also add your file format to the [weio](http://github.com/ebranlard/weio/) repository.
Follow the procedure mentioned in the README of the weio repository (in particualr adding unit tests and minimalistic example files).

!important;width: 174px" ></a>

<a href="https://www.buymeacoffee.com/hTpOQGl" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Donate just a small amount, buy me a coffee" style="height: 41px !important;width: 174px" ></a>
