import importlib
import os
import struct
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

    def test_versioned_flex_outputs_and_legacy_reader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            modern = os.path.join(temp_dir, 'modern.int_2')
            with open(modern, 'wb') as stream:
                stream.write(b'header\xff\xff\xff\xffblock\xff\xff\xff\xff')
                stream.write(struct.pack('<5i', 0, 1, 7, 2, 0))
                for value in (b'Load\x00', b'N\x00', b'OK\x00'):
                    stream.write(struct.pack('<i', len(value)))
                    stream.write(value)
                stream.write(struct.pack('<iff', 0, 1.0, 0.5))
                stream.write(struct.pack('<ff2H', 10.0, 2.0, 3, 4))
            fmt, obj = weio.detectFormat(modern)
            self.assertEqual(fmt.name, 'FLEX output file')
            self.assertEqual(obj.toDataFrame()['Load_[N]'].tolist(), [16.0, 18.0])

            legacy = os.path.join(temp_dir, 'legacy.int_3')
            with open(legacy, 'wb') as stream:
                stream.write(struct.pack('<i6i', 0, 1, 1, 2020, 1, 0, 0))
                stream.write(b'Legacy'.ljust(40, b' '))
                stream.write(struct.pack('<2i', 0, 0))
                stream.write(struct.pack('<3i', 1, 7, 0))
                stream.write(struct.pack('<ifff', 2, 1.0, 0.5, 2.0))
                stream.write(struct.pack('<2h', 3, 4))
            fmt, obj = weio.detectFormat(legacy)
            self.assertEqual(fmt.name, 'FLEX legacy output file')
            self.assertEqual(obj.toDataFrame()['S0001_[NA]'].tolist(), [6.0, 8.0])

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
