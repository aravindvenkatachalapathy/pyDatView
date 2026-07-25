import unittest

import numpy as np

from pydatview.plugins.plotdata_binning import bin_plot, binningAction


class TestBinning(unittest.TestCase):
    def test_headless_action(self):
        opts = {"active": False, "xMin": 0.0, "xMax": 10.0, "nBins": 2}
        action = binningAction(data=opts)
        self.assertIsNone(action.guiEditorClass)
        x_new, y_new = bin_plot(np.arange(10.0), np.arange(10.0), opts)
        self.assertEqual(len(x_new), 2)
        self.assertEqual(y_new.shape, x_new.shape)


if __name__ == "__main__":
    unittest.main()
