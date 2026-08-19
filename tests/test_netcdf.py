import os
import tempfile
import unittest
from unittest.mock import patch

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

    def test_converts_3d_variable_to_grouped_2d_component_slices(self):
        frames = NetCDFFile(self.path).toDataFrame()

        self.assertEqual(set(frames), {
            'speed',
            'field',
            'cube [component=u]',
            'cube [component=v]',
            'reference_speed',
        })
        self.assertEqual(list(frames['speed'].columns), ['time', 'speed'])
        self.assertEqual(
            list(frames['field'].columns), ['time', 'height', 'field']
        )
        self.assertEqual(
            list(frames['cube [component=u]'].columns),
            ['time', 'cube [height=50]', 'cube [height=100]'],
        )
        self.assertEqual(frames['cube [component=u]'].shape, (3, 3))
        self.assertEqual(
            frames['cube [component=u]'].attrs['pydatview']['slice_dimension'],
            'component',
        )
        self.assertEqual(frames['reference_speed'].iloc[0, 0], 9.5)

    def test_loads_through_gui_table_path(self):
        tables, warnings = TableList().load_tables_from_files([self.path])

        self.assertFalse(warnings)
        self.assertEqual(len(tables), 5)
        self.assertTrue(all(
            table.fileformat.name == 'NetCDF file' for table in tables
        ))
        groups = TableList(tables).side_by_side_groups()
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            [table.nickname for table in groups[0]],
            ['cube [component=u]', 'cube [component=v]'],
        )

    def test_3d_variable_without_component_dimension_slices_smallest_axis(self):
        path = os.path.join(self.temp_dir.name, 'volume.nc')
        xr.Dataset(
            data_vars={
                'pressure': (
                    ('x', 'layer', 'y'),
                    np.arange(24.0).reshape(4, 2, 3),
                ),
            },
            coords={
                'x': [0.0, 1.0, 2.0, 3.0],
                'layer': [10, 20],
                'y': [-1.0, 0.0, 1.0],
            },
        ).to_netcdf(path, engine='scipy')

        frames = NetCDFFile(path).toDataFrame()

        self.assertEqual(
            list(frames),
            ['pressure [layer=10]', 'pressure [layer=20]'],
        )
        self.assertEqual(frames['pressure [layer=10]'].shape, (4, 4))
        self.assertEqual(
            frames['pressure [layer=10]'].attrs['pydatview'][
                'slice_dimension'
            ],
            'layer',
        )

    def test_large_3d_variable_loads_plot_columns_on_demand(self):
        with patch.object(NetCDFFile, '_EAGER_3D_LIMIT_BYTES', 1):
            tables, warnings = TableList().load_tables_from_files([self.path])

        self.assertFalse(warnings)
        cube_u = next(
            table for table in tables
            if table.nickname == 'cube [component=u]'
        )
        self.assertTrue(cube_u.source_metadata['lazy_values'])
        self.assertTrue(cube_u.data.iloc[:, 2].isna().all())

        values, is_string, is_date, _series = cube_u.getColumn(2)

        np.testing.assert_array_equal(values, [0.0, 4.0, 8.0])
        self.assertFalse(is_string)
        self.assertFalse(is_date)
        self.assertEqual(
            cube_u._native_plot_backend,
            'xarray lazy NetCDF',
        )


if __name__ == '__main__':
    unittest.main()
