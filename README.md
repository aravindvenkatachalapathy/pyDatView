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

Use **Tools > Standardize units...** and choose **Wind Energy / OpenFAST** to
convert loaded channels to common wind-energy display units, including `Nm`
to `kNm`, `N` to `kN`, `W` to `kW`, `rad` to `deg`, and `rad/s` to `rpm`.
Choose **SI** in the same dialog to convert these channels back to SI units.

Formats without lightweight channel headers use the normal full-file loader.

Enable **Keep files from previous scans** in the scan dialog to append new
matches to the current index. Existing loaded data, cached variables, and the
current file selection are retained; duplicate paths are ignored.

## Plot Navigation

Moving the pointer over a plot shows its X and Y coordinates in the status
bar. Enable **Zoom area** and drag a rectangle over the required plot or
subplot. Use **View > Auto range** to restore the complete data range.
Use **View > Increase font size** or **Decrease font size** to adjust interface
text without changing plot-axis typography.

Interactive plots use Matplotlib's default Tableau color cycle and familiar
white-background axis styling while retaining PyQtGraph's responsiveness for
large time series.

The Plot selector includes **Compare**, with Relative, absolute-relative,
Ratio, Absolute, and Y-Y comparisons against the first selected series in
each group. Comparison legends show `candidate file - reference file` so the
direction is explicit. **Swap X-Y** exchanges the displayed axes after the
selected plot transformation, including Compare, FFT, PDF, and MinMax.

Select a numeric variable in one or more files and choose **Box Plot** to show
one distribution box per file. The X-axis identifies the files; the Y-axis is
the selected variable. Each box spans the first and third quartiles, shows the
median and mean, and uses the full data minimum and maximum as its whiskers.

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

## Publication Export

Use **File > Export publication plot** to render the current transformed plot
with Matplotlib without changing the interactive PyQtGraph view. Available
formats are vector PDF and SVG, LaTeX PGF, and PNG or TIFF up to 1200 DPI.
The dialog controls physical figure dimensions, typography, line width,
peak-preserving vector point limits, grid, legend, transparency, and editable
X-axis and Y-axis labels. Export legends use compact `Set 1`, `Set 2`
labels by default; each label can be renamed in the dialog before export.
Original source names remain available as tooltips in the legend-label table.

PGF and the optional **Use LaTeX text** setting require a LaTeX distribution
on the exporting machine. PDF and SVG remain publication-quality vector
options without LaTeX. For very large channels, the exporter retains local
minima and maxima while reducing the number of vector path points.

## Tests

Run the full retained application and numerical test suite:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

For a headless Linux or macOS session:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

For headless Windows PowerShell:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q
```

## Code Layout

- `qt_main.py` builds the main window, menus, and shared controls.
- `qt_loading.py` owns scanning, lazy loading, and table indexing.
- `qt_selection.py` owns column selection and plot-data construction.
- `qt_tools.py` owns calculations, units, statistics, and exports.
- `qt_plot.py` owns the PyQtGraph canvas and plot styling.
- `qt_dialogs.py` owns standalone dialogs and table models.
- `qt_io.py`, `qt_stats.py`, and `qt_math.py` contain reusable non-window helpers.
- `Tables.py`, `plotdata.py`, `io/`, and `tools/` provide the shared data model,
  readers, and numerical routines.

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
  --collect-all matplotlib `
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

Implement a subclass of `pydatview.io.file.File`, following the readers under
`pydatview/io`, and register its `FileFormat` in `pydatview/io/__init__.py`.
