import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from pydatview.io.bladed_out_file import BladedFile, OrgData
from pydatview.io.load_estimates import estimate_decoded_load_bytes
from pydatview.Tables import Table, TableList
from pydatview.io.file_formats import FileFormat


class TestBladedLazyProject(unittest.TestCase):

    def test_three_dimensional_flattening_is_a_dtype_preserving_view(self):
        values = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
        flattened, _info = OrgData(
            values,
            NDIMENS=3,
            nMajor=5,
            SectionList=np.array([0.0, 1.0, 2.0]),
            ChannelName=['A', 'B'],
            ChannelUnit=['N', 'N'],
        )

        self.assertEqual(flattened.dtype, np.float32)
        self.assertTrue(np.shares_memory(flattened, values))
        self.assertEqual(flattened.shape, (5, 6))

    @staticmethod
    def _write_project_dataset(directory, dimensions, values, category):
        project = os.path.join(directory, 'case.$PJ')
        sensor = os.path.join(directory, 'case.%01')
        binary = os.path.join(directory, 'case.$01')
        with open(project, 'w', encoding='ascii') as stream:
            stream.write('Bladed project\n')
        n_major, n_sections, n_sensors = dimensions
        with open(sensor, 'w', encoding='ascii') as stream:
            stream.write('NDIMENS {}\n'.format(3 if n_sections > 1 else 2))
            if n_sections > 1:
                stream.write(
                    'DIMENS {} {} {}\n'.format(
                        n_sensors,
                        n_sections,
                        n_major,
                    )
                )
                stream.write(
                    'AXIVAL {}\n'.format(
                        ' '.join(str(index) for index in range(n_sections))
                    )
                )
            else:
                stream.write('DIMENS {} {}\n'.format(n_sensors, n_major))
            stream.write('FORMAT R*4\n')
            stream.write("GENLAB '{}'\n".format(category))
            stream.write(
                "VARIAB {}\n".format(
                    ' '.join("'Channel {}'".format(i) for i in range(n_sensors))
                )
            )
            stream.write(
                "VARUNIT {}\n".format(
                    ' '.join("'N'" for _ in range(n_sensors))
                )
            )
            stream.write('MIN 0\n')
            stream.write('STEP 0.25\n')
        np.asarray(values, dtype=np.float32).tofile(binary)
        return project

    def test_project_binary_is_metadata_only_until_channel_access(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(10, dtype=np.float32).reshape(5, 1, 2)
            project = self._write_project_dataset(
                directory,
                (5, 1, 2),
                values,
                'Control variables',
            )

            bladed = BladedFile(project)
            dataset = bladed.dataSets['Control variables']
            frame = bladed.toDataFrame()
            matrix, offset, backend = bladed.get_numpy_plot_data(
                'Control variables'
            )

            self.assertIsNone(dataset['data'])
            self.assertEqual(offset, 2)
            self.assertEqual(backend, 'Bladed memmap')
            self.assertIsNone(matrix._memmap)
            self.assertEqual(frame.iloc[:, 1].array.sp_values.size, 0)
            np.testing.assert_allclose(matrix[:, 1], values[:, 0, 1])
            self.assertIsNotNone(matrix._memmap)

    def test_three_dimensional_channels_are_flattened_as_views(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
            project = self._write_project_dataset(
                directory,
                (5, 3, 2),
                values,
                'Blade loads',
            )

            bladed = BladedFile(project)
            frame = bladed.toDataFrame()
            matrix, _offset, _backend = bladed.get_numpy_plot_data(
                'Blade loads'
            )

            self.assertEqual(matrix.shape, (5, 6))
            self.assertEqual(frame.shape, (5, 7))
            np.testing.assert_allclose(matrix[:, 4], values[:, 2, 0])

    def test_project_memory_estimate_reflects_lazy_working_set(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(30, dtype=np.float32).reshape(5, 3, 2)
            project = self._write_project_dataset(
                directory,
                (5, 3, 2),
                values,
                'Blade loads',
            )

            estimated = estimate_decoded_load_bytes(
                project,
                SimpleNamespace(name='Bladed output file'),
            )

            self.assertEqual(estimated, 5 * 24)

    def test_unit_conversion_preserves_lazy_channel_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(10, dtype=np.float32).reshape(5, 1, 2)
            project = self._write_project_dataset(
                directory,
                (5, 1, 2),
                values,
                'Control variables',
            )
            bladed = BladedFile(project)
            table = Table(
                data=bladed.toDataFrame(),
                name='Control variables',
                filename=project,
                fileobject=bladed,
            )

            table.changeUnits(data={'flavor': 'WE'})
            converted, _is_string, _is_date, _series = table.getColumn(2)

            self.assertIsNotNone(table._native_plot_matrix)
            self.assertIn('Channel 0 [kN]', table.data.columns)
            np.testing.assert_allclose(converted, values[:, 0, 0] * 1e-3)
            self.assertEqual(table.data.iloc[:, 2].array.sp_values.size, 0)

    def test_table_loader_keeps_bladed_project_channels_on_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            values = np.arange(10, dtype=np.float32).reshape(5, 1, 2)
            project = self._write_project_dataset(
                directory,
                (5, 1, 2),
                values,
                'Control variables',
            )
            tables = TableList()

            loaded, warning = tables._load_file_tabs(
                project,
                fileformat=FileFormat(BladedFile),
            )

            self.assertEqual(warning, '')
            self.assertEqual(len(loaded), 1)
            self.assertIsNotNone(loaded[0]._native_plot_matrix)
            channel, *_rest = loaded[0].getColumn(2)
            np.testing.assert_allclose(channel, values[:, 0, 0])


if __name__ == '__main__':
    unittest.main()
