import pandas as pd

from .file import BrokenFormatError, File, OptionalImportError


class NetCDFFile(File):

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

    def _toDataFrame(self):
        """Return one value-preserving, tabular view for each data variable.

        xarray supplies dimension coordinates as columns. Keeping variables in
        separate frames avoids broadcasting unrelated variables onto the full
        Cartesian product of every dimension in the dataset.
        """
        dfs = {}
        for name, variable in self.data.data_vars.items():
            if variable.ndim == 0:
                dfs[name] = pd.DataFrame({name: [variable.item()]})
            else:
                dfs[name] = variable.to_dataframe(name=name).reset_index()
        return dfs
