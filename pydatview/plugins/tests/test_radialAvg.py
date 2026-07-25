import unittest

from pydatview.plugins.data_radialavg import radialAvg, radialAvgAction


class TestRadialAvg(unittest.TestCase):
    def test_headless_action(self):
        action = radialAvgAction("radial average")
        self.assertIsNone(action.guiEditorClass)
        self.assertIs(action.tableFunctionAdd, radialAvg)
        self.assertIsNone(action.guiCallback)


if __name__ == "__main__":
    unittest.main()
