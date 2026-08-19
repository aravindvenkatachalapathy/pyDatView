import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from .file import BrokenFormatError, File, OptionalImportError


class _NetCDFSliceMatrix:
    """Array-like, column-oriented view of one NetCDF 2-D slice."""

    ndim = 2

    def __init__(
            self,
            variable,
            slice_dimension,
            slice_index,
            row_dimension,
            column_dimension,
            cache_size=4):
        self.variable = variable
        self.slice_dimension = slice_dimension
        self.slice_index = slice_index
        self.row_dimension = row_dimension
        self.column_dimension = column_dimension
        self.shape = (
            variable.sizes[self.row_dimension],
            variable.sizes[self.column_dimension],
        )
        self.cache_size = cache_size
        self._column_cache = OrderedDict()

    def _column(self, column_index):
        column_index = int(column_index)
        cached = self._column_cache.pop(column_index, None)
        if cached is not None:
            self._column_cache[column_index] = cached
            return cached
        values = np.asarray(self.variable.isel({
            self.slice_dimension: self.slice_index,
            self.column_dimension: column_index,
        }).values)
        self._column_cache[column_index] = values
        while len(self._column_cache) > self.cache_size:
            self._column_cache.popitem(last=False)
        return values

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError('NetCDF slice data requires row and column indices')
        row_selector, column_selector = key
        if not np.isscalar(column_selector):
            raise IndexError('NetCDF slice data is loaded one column at a time')
        return self._column(column_selector)[row_selector]


class NetCDFFile(File):

    _EAGER_3D_LIMIT_BYTES = 128 * 1024 * 1024

    _COMPONENT_DIMENSION_NAMES = {
        'axis',
        'comp',
        'component',
        'components',
        'direction',
        'directions',
        'vector',
    }

    @staticmethod
    def defaultExtensions():
        return ['.nc', '.nc4', '.cdf']

    @staticmethod
    def formatName():
        return 'NetCDF file'

    def _read(self, **kwargs):
        try:
            import xarray as xr
        except ImportError as error:
            raise OptionalImportError(
                'Install the library xarray to read NetCDF files'
            ) from error

        try:
            kwargs.setdefault('cache', False)
            self.data = xr.open_dataset(self.filename, **kwargs)
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalImportError(
                'No NetCDF backend is available. Install the library netCDF4.'
            ) from error
        except ValueError as error:
            # xarray uses ValueError when none of its installed engines can
            # handle a file, which most often means a NetCDF4/HDF5 file was
            # opened without netCDF4 or h5netcdf installed.
            if 'installed IO backends' in str(error):
                raise OptionalImportError(
                    'No installed xarray backend can read this NetCDF file. '
                    'Install the library netCDF4.'
                ) from error
            raise BrokenFormatError(
                'xarray could not read the NetCDF file: {}'.format(error)
            ) from error
        except OSError as error:
            raise BrokenFormatError(
                'xarray could not read the NetCDF file: {}'.format(error)
            ) from error

    def _write(self, **kwargs):
        self.data.to_netcdf(self.filename, **kwargs)

    @staticmethod
    def _dimension_values(variable, dimension):
        coordinate = variable.coords.get(dimension)
        if coordinate is not None and coordinate.dims == (dimension,):
            return np.asarray(coordinate.values)
        return np.arange(variable.sizes[dimension])

    @staticmethod
    def _coordinate_text(value):
        if isinstance(value, bytes):
            return value.decode(errors='replace')
        if isinstance(value, (float, np.floating)):
            return '{:.7g}'.format(float(value))
        return str(value)

    @classmethod
    def _slice_dimension(cls, variable):
        for dimension in variable.dims:
            normalized = str(dimension).lower()
            if normalized in cls._COMPONENT_DIMENSION_NAMES:
                return dimension
        return min(variable.dims, key=lambda dim: variable.sizes[dim])

    @classmethod
    def _orient_two_dimensional_variable(cls, variable):
        first_dimension, second_dimension = variable.dims
        if variable.sizes[first_dimension] < variable.sizes[second_dimension]:
            return variable.transpose(second_dimension, first_dimension)
        return variable

    @classmethod
    def _two_dimensional_frame(cls, name, variable, load_values=True):
        variable = cls._orient_two_dimensional_variable(variable)
        row_dimension, column_dimension = variable.dims
        row_values = cls._dimension_values(variable, row_dimension)
        column_values = cls._dimension_values(variable, column_dimension)
        matrix = np.asarray(variable.values) if load_values else None

        columns = {str(row_dimension): row_values}
        used_columns = {str(row_dimension)}
        placeholder = None
        if not load_values:
            placeholder = pd.arrays.SparseArray(
                np.full(len(row_values), np.nan),
                fill_value=np.nan,
            )
        for column_index, coordinate in enumerate(column_values):
            column = '{} [{}={}]'.format(
                name,
                column_dimension,
                cls._coordinate_text(coordinate),
            )
            if column in used_columns:
                column = '{} [index={}]'.format(column, column_index)
            used_columns.add(column)
            if load_values:
                columns[column] = matrix[:, column_index]
            else:
                columns[column] = placeholder
        return pd.DataFrame(columns)

    def _lazy_3d_variable(self, variable):
        try:
            numeric = np.issubdtype(variable.dtype, np.number)
        except TypeError:
            numeric = False
        if not numeric:
            return False
        return (
            self.size >= self._EAGER_3D_LIMIT_BYTES
            or variable.nbytes >= self._EAGER_3D_LIMIT_BYTES
        )

    def _three_dimensional_frames(self, name, variable):
        slice_dimension = self._slice_dimension(variable)
        slice_values = self._dimension_values(variable, slice_dimension)
        group = ('netcdf-3d', os.path.abspath(self.filename), str(name))
        lazy_values = self._lazy_3d_variable(variable)
        frames = {}
        for slice_index, coordinate in enumerate(slice_values):
            plane = variable.isel({slice_dimension: slice_index}, drop=True)
            plane = self._orient_two_dimensional_variable(plane)
            key = '{} [{}={}]'.format(
                name,
                slice_dimension,
                self._coordinate_text(coordinate),
            )
            if key in frames:
                key = '{} [index={}]'.format(key, slice_index)
            frame = self._two_dimensional_frame(
                name,
                plane,
                load_values=not lazy_values,
            )
            frame.attrs['pydatview'] = {
                'side_by_side_group': group,
                'lazy_values': lazy_values,
                'lazy_column_offset': 2,
                'slice_dimension': str(slice_dimension),
                'slice_index': slice_index,
                'slice_value': self._coordinate_text(coordinate),
                'source_variable': str(name),
            }
            frames[key] = frame
            if lazy_values:
                self._native_plot_sources[key] = (
                    _NetCDFSliceMatrix(
                        variable,
                        slice_dimension,
                        slice_index,
                        plane.dims[0],
                        plane.dims[1],
                    ),
                    2,
                    'xarray lazy NetCDF',
                )
        return frames

    def get_numpy_plot_data(self, table_name=''):
        return getattr(self, '_native_plot_sources', {}).get(table_name)

    def _toDataFrame(self):
        """Return value-preserving tabular views for the data variables.

        xarray supplies dimension coordinates as columns. Keeping variables in
        separate frames avoids broadcasting unrelated variables onto the full
        Cartesian product of every dimension in the dataset. Three-dimensional
        variables become grouped two-dimensional slice tables so the GUI can
        display sibling slices in side-by-side selector panes.
        """
        self._native_plot_sources = {}
        dfs = {}
        for name, variable in self.data.data_vars.items():
            if variable.ndim == 0:
                dfs[name] = pd.DataFrame({name: [variable.item()]})
            elif variable.ndim == 3:
                dfs.update(self._three_dimensional_frames(name, variable))
            else:
                dfs[name] = variable.to_dataframe(name=name).reset_index()
        return dfs
