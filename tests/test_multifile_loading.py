import os
import tempfile
import unittest

import pandas as pd

from pydatview.Tables import Table, TableList


class TestMultiFileLoadingSafety(unittest.TestCase):
    def test_relative_and_absolute_paths_do_not_load_the_same_file_twice(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as folder:
            path = os.path.join(folder, "case.csv")
            pd.DataFrame({"Time": [0.0], "Load": [1.0]}).to_csv(
                path, index=False
            )
            relative_path = os.path.relpath(path)
            tables = TableList()

            added, warnings = tables.load_tables_from_files(
                [path, relative_path], bAdd=False
            )

            self.assertEqual(len(added), 1)
            self.assertEqual(len(tables), 1)
            self.assertEqual(len(warnings), 1)
            self.assertIn("already opened", warnings[0])

    def test_mismatched_format_list_does_not_clear_existing_tables(self):
        tables = TableList()
        existing = Table(data=pd.DataFrame({"Value": [1.0]}), name="existing")
        tables.append(existing)

        with self.assertRaisesRegex(ValueError, "one entry per filename"):
            tables.load_tables_from_files(
                ["first.out", "second.out"],
                fileformats=[None],
                bAdd=False,
            )

        self.assertEqual(len(tables), 1)
        self.assertIs(tables[0], existing)


if __name__ == "__main__":
    unittest.main()
