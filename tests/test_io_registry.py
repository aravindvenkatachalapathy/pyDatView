import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

import pandas as pd
from scipy.io import savemat

import pydatview.io as weio
from pydatview.Tables import TableList
from pydatview.io.excel_file import ExcelFile
from pydatview.io.file import EmptyFileError


class TestIORegistry(unittest.TestCase):
    @staticmethod
    def builtin_formats():
        return [
            file_format for file_format in weio.fileFormats()
            if file_format.constructor.__module__.startswith('pydatview.io.')
        ]

    def test_all_registered_readers_share_empty_file_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for file_format in self.builtin_formats():
                extension = next(
                    (
                        item for item in file_format.extensions
                        if item.startswith('.') and '*' not in item
                    ),
                    '.tmp',
                )
                path = os.path.join(temp_dir, 'empty' + extension)
                with open(path, 'wb'):
                    pass
                with self.subTest(reader=file_format.constructor.__name__):
                    with self.assertRaises(EmptyFileError):
                        file_format.constructor(filename=path)

    def test_registered_extensions_are_normalized(self):
        for file_format in self.builtin_formats():
            for extension in file_format.extensions:
                with self.subTest(
                    reader=file_format.constructor.__name__,
                    extension=extension,
                ):
                    self.assertTrue(extension.startswith('.'))
                    self.assertEqual(extension, extension.lower())

    def test_numbered_flex_profile_extension_is_detected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'profile.001')
            with open(path, 'w', encoding='ascii') as stream:
                stream.write(
                    'profile\n1\n20\n2\npolar\n'
                    '0 0 0 0\n1 1 1 1\n'
                )

            file_format, file_object = weio.detectFormat(path)
            self.assertEqual(file_format.name, 'FLEX profile file')
            self.assertEqual(len(file_object.toDataFrame()), 1)

    def test_generic_matlab_file_loads_after_raaw_probe(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'signals.mat')
            savemat(path, {
                'time': [0.0, 1.0, 2.0],
                'loads': [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            })

            tables, warnings = TableList().load_tables_from_files([path])
            self.assertFalse(warnings)
            self.assertEqual(len(tables), 2)
            self.assertTrue(all(
                table.fileformat.name == 'MATLAB MAT file' for table in tables
            ))

    def test_excel_xlsx_round_trip_uses_declared_backend(self):
        expected = pd.DataFrame({
            'time': [0, 1],
            'load': [2.0, 3.0],
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'signals.xlsx')
            workbook = ExcelFile()
            workbook.data = {'signals': expected}
            workbook.write(path)
            actual = ExcelFile(path).toDataFrame()

        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

    def test_legacy_user_module_uses_local_io_package(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            weio, 'defaultUserDataDir', return_value=temp_dir
        ):
            sys.modules.pop('pydatview.io.user', None)
            module = importlib.import_module('pydatview.io.user')
        self.assertTrue(hasattr(module, 'UserClasses'))


if __name__ == '__main__':
    unittest.main()
