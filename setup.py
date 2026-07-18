from setuptools import setup

setup(
    name='pydatview',
    version='0.5',
    description='GUI to display tabulated data from files or pandas dataframes',
    url='http://github.com/ebranlard/pyDatView/',
    author='Emmanuel Branlard',
    author_email='lastname@gmail.com',
    license='MIT',
    packages=['pydatview'],
    install_requires=[
        'openpyxl',
        'numpy',
        'pandas',
        'xarray',
        'pyarrow',
        'matplotlib',
        'chardet',
        'scipy',
        'PySide6',
        'pyqtgraph',
    ],
    extras_require={
        'legacy_wx': ['wxpython'],
    },
    zip_safe=False
)
