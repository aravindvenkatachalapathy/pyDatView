import unittest
import numpy as np
import pandas as pd
from pydatview.plugins.data_standardizeUnits import changeUnitsTab

class TestChangeUnits(unittest.TestCase):

    def test_change_units(self):
        # TODO move this as a pandalib test
        from pydatview.Tables import Table
        data = np.ones((1,6))
        data[:,0] *= 2*np.pi/60    # rad/s
        data[:,1] *= 2000          # N
        data[:,2] *= 10*np.pi/180  # rad
        data[:,3] *= 2500           # Nm
        data[:,4] *= 2e6            # W
        data[:,5] *= 3500           # N-m
        df = pd.DataFrame(
            data=data,
            columns=[
                'om [rad/s]', 'F [N]', 'angle_[rad]', 'Torque [Nm]',
                'Power [W]', 'Moment [N-m]',
            ],
        )
        tab=Table(data=df)
        changeUnitsTab(tab, data={'flavor':'WE'})
        np.testing.assert_almost_equal(tab.data.values[:,1],[1])
        np.testing.assert_almost_equal(tab.data.values[:,2],[2])
        np.testing.assert_almost_equal(tab.data.values[:,3],[10])
        np.testing.assert_almost_equal(tab.data.values[:,4],[2.5])
        np.testing.assert_almost_equal(tab.data.values[:,5],[2000])
        np.testing.assert_almost_equal(tab.data.values[:,6],[3.5])
        np.testing.assert_equal(
            list(tab.columns),
            [
                'Index', 'om [rpm]', 'F [kN]', 'angle_[deg]',
                'Torque [kNm]', 'Power [kW]', 'Moment [kNm]',
            ],
        )


if __name__ == '__main__':
    unittest.main()
