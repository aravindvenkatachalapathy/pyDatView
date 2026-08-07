from setuptools import find_packages, setup

setup(
    name='pydatview',
    version='0.5',
    description='Qt GUI to load, compare, and plot engineering data',
    url='https://github.com/aravindvenkatachalapathy/pyDatView',
    author='Aravind Venkatachalapathy',
    author_email='lastname@gmail.com',
    license='MIT',
    python_requires='>=3.9',
    packages=find_packages(include=['pydatview', 'pydatview.*']),
    data_files=[
        ('ressources', ['ressources/pyDatView.ico']),
        (
            'ressources/icons',
            [
                'ressources/icons/chart.svg',
                'ressources/icons/filesave.svg',
                'ressources/icons/scan.png',
            ],
        ),
    ],
    install_requires=[
        'openpyxl',
        'numpy',
        'pandas',
        'polars',
        'xarray',
        'pyarrow',
        'matplotlib',
        'chardet',
        'scipy',
        'PySide6',
        'pyqtgraph',
        'psutil',
    ],
    extras_require={
        'build': ['pyinstaller>=6'],
        'rust-build': ['maturin>=1.4,<2'],
        'test': ['pytest>=7'],
    },
    entry_points={
        'gui_scripts': [
            'pydatview=pydatview.qt_main:cmdline',
        ],
    },
    zip_safe=False
)
