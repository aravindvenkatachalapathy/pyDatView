import unittest

import numpy as np
import pandas as pd

from pydatview.Tables import Table, TableList
from pydatview.qt_math import evaluate_table_script, transform_file_tables


class TestTableTransforms(unittest.TestCase):

    def setUp(self):
        self.dataframe = pd.DataFrame({
            'Index': np.arange(6),
            'Time_[s]': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'Power_[kW]': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            'Speed_[rpm]': [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        })

    def test_trim_auto_detects_time_and_keeps_all_variables(self):
        result, positions = evaluate_table_script(
            self.dataframe,
            'trim(start=1, stop=3)',
        )

        np.testing.assert_array_equal(positions, [1, 2, 3])
        self.assertEqual(list(result.columns), list(self.dataframe.columns))
        np.testing.assert_array_equal(result['Index'], [0, 1, 2])
        np.testing.assert_array_equal(result['Power_[kW]'], [11.0, 12.0, 13.0])
        np.testing.assert_array_equal(result['Speed_[rpm]'], [21.0, 22.0, 23.0])

    def test_trim_accepts_explicit_column_and_open_bound(self):
        result, positions = evaluate_table_script(
            self.dataframe,
            'trim(x={Time_[s]}, start=3)',
        )

        np.testing.assert_array_equal(positions, [3, 4, 5])
        np.testing.assert_array_equal(result['Time_[s]'], [3.0, 4.0, 5.0])

    def test_transform_script_rejects_unregistered_python(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported table transform'):
            evaluate_table_script(self.dataframe, '__import__(name="os")')

    def test_trim_accepts_datetime_bounds(self):
        dataframe = pd.DataFrame({
            'Time': pd.date_range('2025-01-01', periods=4, freq='h'),
            'Value': [10.0, 20.0, 30.0, 40.0],
        })

        result, positions = evaluate_table_script(
            dataframe,
            'trim(start="2025-01-01 01:00", stop="2025-01-01 02:00")',
        )

        np.testing.assert_array_equal(positions, [1, 2])
        np.testing.assert_array_equal(result['Value'], [20.0, 30.0])

    def test_file_transform_trims_every_time_table_and_copies_static_tables(self):
        filename = '/tmp/source.nc'
        tables = TableList([
            Table(
                data=self.dataframe.copy(),
                name='signals_a',
                filename=filename,
            ),
            Table(
                data=self.dataframe.rename(
                    columns={'Power_[kW]': 'Load_[kN]'}
                ).copy(),
                name='signals_b',
                filename=filename,
            ),
            Table(
                data=pd.DataFrame({'Reference': [12.0]}),
                name='metadata',
                filename=filename,
            ),
            Table(
                data=self.dataframe.copy(),
                name='other_file',
                filename='/tmp/other.nc',
            ),
        ])

        transformed, indices, trimmed, static = transform_file_tables(
            tables,
            0,
            '_trimmed',
            'trim(start=2, stop=4)',
        )

        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(trimmed, 2)
        self.assertEqual(static, 1)
        self.assertEqual(len(transformed), 3)
        np.testing.assert_array_equal(
            transformed[0].data['Power_[kW]'],
            [12.0, 13.0, 14.0],
        )
        np.testing.assert_array_equal(
            transformed[1].data['Load_[kN]'],
            [12.0, 13.0, 14.0],
        )
        np.testing.assert_array_equal(
            transformed[2].data['Reference'],
            [12.0],
        )


if __name__ == '__main__':
    unittest.main()
