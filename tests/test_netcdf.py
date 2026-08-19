import os
import tempfile
import unittest

import numpy as np
import xarray as xr

from pydatview.Tables import TableList
from pydatview.io import detectFormat
from pydatview.io.netcdf_file import NetCDFFile


class TestNetCDFFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, 'signals.nc')
        dataset = xr.Dataset(
            data_vars={
                'speed': ('time', [8.0, 9.0, 10.0]),
                'field': (
                    ('time', 'height'),
                    np.arange(6.0).reshape(3, 2),
                ),
                'cube': (
                    ('time', 'height', 'component'),
                    np.arange(12.0).reshape(3, 2, 2),
                ),
                'reference_speed': 9.5,
            },
            coords={
                'time': [0.0, 1.0, 2.0],
                'height': [50.0, 100.0],
                'component': ['u', 'v'],
            },
        )
        # SciPy writes NetCDF3, so this test remains independent of the
        # NetCDF4 backend while exercising the same xarray reader path.
        dataset.to_netcdf(self.path, engine='scipy')

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_detects_common_netcdf_extensions(self):
        self.assertEqual(
            NetCDFFile.defaultExtensions(), ['.nc', '.nc4', '.cdf']
        )
        file_format, file_object = detectFormat(self.path)
        self.assertEqual(file_format.name, 'NetCDF file')
        self.assertIsInstance(file_object, NetCDFFile)

    def test_converts_coordinates_and_all_variable_dimensions(self):
        frames = NetCDFFile(self.path).toDataFrame()

        self.assertEqual(set(frames), {
            'speed', 'field', 'cube', 'reference_speed'
        })
        self.assertEqual(list(frames['speed'].columns), ['time', 'speed'])
        self.assertEqual(
            list(frames['field'].columns), ['time', 'height', 'field']
        )
        self.assertEqual(
            list(frames['cube'].columns),
            ['time', 'height', 'component', 'cube'],
        )
        self.assertEqual(frames['cube'].shape, (12, 4))
        self.assertEqual(frames['reference_speed'].iloc[0, 0], 9.5)

    def test_loads_through_gui_table_path(self):
        tables, warnings = TableList().load_tables_from_files([self.path])

        self.assertFalse(warnings)
        self.assertEqual(len(tables), 4)
        self.assertTrue(all(
            table.fileformat.name == 'NetCDF file' for table in tables
        ))


if __name__ == '__main__':
    unittest.main()
