# pyDatView

pyDatView is a desktop application for loading, comparing, and plotting
engineering data. The GUI uses PySide6 and PyQtGraph. Large OpenFAST and
Bladed binary files can use an optional Rust extension.

Python 3.9 and newer is supported. Python 3.13 is the recommended reproducible
version for new environments.

## Clone

Install Git and Python or conda, then clone the repository:

```bash
git clone https://github.com/aravindvenkatachalapathy/pyDatView.git
cd pyDatView
```

## Install With Conda

This is the simplest identical setup on Windows, macOS, and Linux:

```bash
conda env create -f environment.yml
conda activate pydatview
```

To update an existing environment after pulling changes:

```bash
conda activate pydatview
python -m pip install -e .
```

## Install With venv

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

If script activation is blocked, run this once in the current PowerShell
session:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### macOS and Linux

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

On Debian or Ubuntu, Qt may require these system libraries:

```bash
sudo apt update
sudo apt install libegl1 libgl1 libxkbcommon-x11-0 libxcb-cursor0
```

## Run

Start the installed application:

```bash
pydatview
```

Equivalent source-tree commands are:

```bash
python -m pydatview
python pyDatView.py
```

Files can be opened from the command line:

```bash
python -m pydatview simulation.outb controller.csv
```

## Optional Rust Fast I/O

Rust accelerates full OpenFAST `.outb` and Bladed binary loading. Python and
NumPy readers remain available when the extension is not installed.

### Conda

```bash
conda activate pydatview
conda install -c conda-forge rust maturin
cd rust/pydatview_fastio
maturin develop --release
cd ../..
```

### venv

Install the Rust toolchain from [rustup.rs](https://rustup.rs/), reactivate the
Python environment, then run:

```bash
python -m pip install "maturin>=1.4,<2"
cd rust/pydatview_fastio
maturin develop --release
cd ../..
```

Verify the extension:

```bash
python -c "import pydatview_fastio; print('Rust fast I/O available')"
```

The extension must be compiled separately on every operating system and for
every Python environment. Do not copy a compiled extension between Windows,
macOS, and Linux.

Bladed batches use a separate concurrency safety cap (two workers on Windows)
and check available memory before starting each project. This cap applies even
when the global worker selector is higher. The Rust reader preserves float32
Bladed data at native precision to reduce retained memory; rebuild the Rust
extension after updating pyDatView to receive this change.

## Scan and Selective Loading

**Scan folder** creates a lightweight file index. For OpenFAST `.outb` and
ASCII output, pressing **Plot** reads only the selected X and Y variables.
The file is shown as `partial N/M`, where `N` is the number of cached variables
and `M` is the number available in the file.

Selecting another variable expands the cache only by the variables required
for the plot. Use **Load full selected** when a complete dataframe is required
for calculations, unit conversion, table preview, or table export.

Use **Tools > Standardize units > Wind Energy / OpenFAST units** to convert
loaded channels to common wind-energy display units, including `Nm` to `kNm`,
`N` to `kN`, `W` to `kW`, `rad` to `deg`, and `rad/s` to `rpm`. The SI action
in the same menu converts these channels back to SI units.

Formats without lightweight channel headers use the normal full-file loader.

Enable **Keep files from previous scans** in the scan dialog to append new
matches to the current index. Existing loaded data, cached variables, and the
current file selection are retained; duplicate paths are ignored.

## Plot Navigation

Moving the pointer over a plot shows its X and Y coordinates in the status
bar. Enable **Zoom area** and drag a rectangle over the required plot or
subplot. Use **View > Auto range** to restore the complete data range.

The Plot selector includes **Compare**, with Relative, absolute-relative,
Ratio, Absolute, and Y-Y comparisons against the first selected series in
each group. Comparison legends show `candidate file - reference file` so the
direction is explicit. **Swap X-Y** exchanges the displayed axes after the
selected plot transformation, including Compare, FFT, PDF, and MinMax.

## Statistics and FFT

The **Stats** tab shows one row per plotted time series. Use its **Columns**
menu to select channel, file, directory, table, sample count, median sample
spacing (`dt`), median, mean, standard deviation, variance, `Std/Mean`, extrema,
X locations of extrema, absolute maximum, X/Y ranges, and integral statistics.
Selections are remembered between sessions. Use the separate **DEL slopes**
menu to add one or more 1 Hz damage-equivalent-load columns for Wohler slopes
from 2 through 13.

Selecting **FFT** opens spectrum controls for PSD, frequency-weighted PSD, or
amplitude; averaging and window selection; Welch segment length or logarithmic
binning resolution; frequency, cyclic-frequency, or period axes; and
detrending. FFT plots enable the base-10 logarithmic Y axis by default and
restore the previous Y-axis mode when returning to a regular plot.

## Tests

Run the core and plugin tests:

```bash
python -m unittest discover -v tests
python -m unittest discover -v pydatview/plugins/tests
```

For a headless Linux or macOS session:

```bash
QT_QPA_PLATFORM=offscreen python -m unittest discover -v tests
```

For headless Windows PowerShell:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m unittest discover -v tests
```

## Build a Windows Executable

Build on Windows using the same environment that will supply the application
dependencies. Build the optional Rust extension first when it should be
included.

```powershell
conda activate pydatview
python -m pip install -e ".[build]"
python -m PyInstaller --noconfirm --clean --windowed --onedir `
  --name pyDatView `
  --icon ressources\pyDatView.ico `
  --add-data "ressources;ressources" `
  --collect-all pyqtgraph `
  pyDatView.py
```

The executable and supporting files are created in `dist\pyDatView`. Test the
entire directory on a clean Windows machine before distributing it.

## Troubleshooting

Confirm that Python and pip refer to the same environment:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

If Python loads packages from a user installation instead of the active
environment, disable user-site packages for the current command:

```bash
PYTHONNOUSERSITE=1 python -m pydatview
```

PowerShell equivalent:

```powershell
$env:PYTHONNOUSERSITE = "1"
python -m pydatview
```

## Adding File Formats

Implement a subclass of `pydatview.io.File.File`, following the readers under
`pydatview/io`, and register its `FileFormat` in `pydatview/io/__init__.py`.
