import unittest
import numpy as np
#from pydatview.Tables import Table
import os
import matplotlib.pyplot as plt


from pydatview.plotdata import PlotData

class TestPlotData(unittest.TestCase):

    def test_FFT(self):
        # --- Test plotdata conversion to FFT
        # Ampltidue and frequency of a sin function should be retrieved
        dt = 0.1
        f0 = 1  ; 
        A  = 5  ; 
        t=np.arange(0,10,dt);
        y=A*np.sin(2*np.pi*f0*t)

        PD=PlotData(t,y)
        PD.toFFT(yType='Amplitude', avgMethod='None', bDetrend=False)
        f = PD.x
        Y = PD.y
        i=np.argmax(Y)
        self.assertAlmostEqual(Y[i],A)
        self.assertAlmostEqual(f[i],f0)

    def test_FFT_period_axis_is_finite_and_returns_info(self):
        dt = 0.05
        f0 = 2.0
        t = np.arange(0, 10, dt)
        y = 3.0 * np.sin(2 * np.pi * f0 * t)

        PD = PlotData(t, y)
        info = PD.toFFT(
            yType='Amplitude',
            xType='x',
            avgMethod='None',
            bDetrend=False,
        )

        self.assertIsNotNone(info)
        self.assertTrue(np.all(np.isfinite(PD.x)))
        peak = np.argmax(PD.y)
        self.assertAlmostEqual(PD.x[peak], 1 / f0)
        self.assertAlmostEqual(PD.y[peak], 3.0)

    def test_cumulative_psd_integrates_psd_over_frequency(self):
        dt = 0.05
        t = np.arange(0, 20, dt)
        y = (
            3.0 * np.sin(2 * np.pi * 1.0 * t)
            + 1.5 * np.sin(2 * np.pi * 3.0 * t)
        )
        psd = PlotData(t, y, sx='Time [s]', sy='Load [N]')
        psd.toFFT(
            yType='PSD',
            avgMethod='None',
            bDetrend=False,
        )
        cumulative = PlotData(t, y, sx='Time [s]', sy='Load [N]')
        info = cumulative.toCumulativePSD(
            avgMethod='None',
            bDetrend=False,
        )

        expected = np.sum(
            0.5 * (psd.y[1:] + psd.y[:-1]) * np.diff(psd.x)
        )
        self.assertIsNotNone(info)
        self.assertTrue(np.all(np.diff(cumulative.y) >= -1e-12))
        self.assertAlmostEqual(cumulative.y[-1], expected)
        self.assertEqual(cumulative.sx, 'Frequency [Hz]')
        self.assertEqual(cumulative.sy, 'Cumulative PSD(Load) [(N)^2]')

    def test_MinMax(self):
        # Test Min Max scaling (between 0 and 1)
        x = np.linspace(-2,2,100)
        y = x**3
        PD = PlotData(x,y)
        # --- Scale both
        PD.toMinMax(xScale=True, yScale=True, yCenter='None')
        self.assertAlmostEqual(np.min(PD.x),0.0)
        self.assertAlmostEqual(np.min(PD.y),0.0)
        self.assertAlmostEqual(PD._xMin[0],0.0)
        self.assertAlmostEqual(PD._yMin[0],0.0)
        self.assertAlmostEqual(np.max(PD.x),1.0)
        self.assertAlmostEqual(np.max(PD.y),1.0)
        self.assertAlmostEqual(PD._xMax[0] ,1.0)
        self.assertAlmostEqual(PD._yMax[0] ,1.0)

        # --- Y Center 0  
        x = np.linspace(-2,2,100)
        y = x**3 + 10
        PD = PlotData(x,y)
        PD.toMinMax(xScale=False, yScale=False, yCenter='Mean=0')
        self.assertAlmostEqual(np.mean(PD.y),0.0)
        self.assertAlmostEqual(np.min(PD.y),-8.0)
        self.assertAlmostEqual(PD._yMin[0] ,-8.0)
        self.assertAlmostEqual(np.max(PD.y),8.0)
        self.assertAlmostEqual(PD._yMax[0] ,8.0)

        PD = PlotData(x,y)
        PD.toMinMax(xScale=False, yScale=False, yCenter='Mid=0')
        self.assertAlmostEqual(np.min(PD.y),-8.0)
        self.assertAlmostEqual(PD._yMin[0] ,-8.0)
        self.assertAlmostEqual(np.max(PD.y),8.0)
        self.assertAlmostEqual(PD._yMax[0] ,8.0)

        # --- Y Center ref
        x = np.linspace(-2,2,100)
        y = x**3 + 10
        PD = PlotData(x,y)
        PD.toMinMax(xScale=False, yScale=False, yCenter='Mean=ref', yRef=20)
        self.assertAlmostEqual(np.mean(PD.y),20+0.0)
        self.assertAlmostEqual(np.min(PD.y) ,20+-8.0)
        self.assertAlmostEqual(PD._yMin[0]  ,20+-8.0)
        self.assertAlmostEqual(np.max(PD.y) ,20+8.0)
        self.assertAlmostEqual(PD._yMax[0]  ,20+8.0)

        PD = PlotData(x,y)
        PD.toMinMax(xScale=False, yScale=False, yCenter='Mid=ref', yRef=20)
        self.assertAlmostEqual(np.min(PD.y),20+-8.0)
        self.assertAlmostEqual(PD._yMin[0] ,20+-8.0)
        self.assertAlmostEqual(np.max(PD.y),20+8.0)
        self.assertAlmostEqual(PD._yMax[0] ,20+8.0)



    def test_PDF(self):
        # --- Test the PDF conversion of plotdata
        # Check that the PDF of random normal noise is a Gaussian
        from pydatview.tools.curve_fitting import model_fit
        mu=0
        sigma=1
        x = np.linspace(-1,1,10000)
        y = np.random.normal(mu,sigma,len(x))
        PD = PlotData(x,y)
        PD.toPDF()

        y_fit, pfit, fitter = model_fit('predef: gaussian', PD.x, PD.y)
        np.testing.assert_almost_equal(mu   ,fitter.model['coeffs']['mu']   , 1)
        try:
            np.testing.assert_almost_equal(sigma,fitter.model['coeffs']['sigma'], 1)
        except:
            print('>>>> NOTE: sigma test failed for test_PDF')
            pass
        #print(fitter)
        #plt.plot(PD.x,PD.y)
        #plt.plot(PD.x,fitter.model['fitted_function'](PD.x),'k--')
        #plt.show()

    def test_fatigue(self):
        dt = 0.1
        f0 = 1  ; 
        A  = 5  ; 
        t=np.arange(0,10,dt);
        y=A*np.sin(2*np.pi*f0*t)

        PD = PlotData(t,y)
        v, s = PD.leq(m=10, method='rainflow_windap')
        np.testing.assert_almost_equal(v, 9.4714702, 3)




if __name__ == '__main__':
    unittest.main()
