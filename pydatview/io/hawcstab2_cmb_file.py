from .file import BrokenFormatError, File, WrongFormatError
import numpy as np
import pandas as pd
from .csv_file import CSVFile


class HAWCStab2CmbFile(File):

    @staticmethod
    def defaultExtensions():
        return ['.cmb']

    @staticmethod
    def formatName():
        return 'HAWCStab2 Campbell file'

    def _read(self):
        try:
            f = CSVFile(self.filename, detectColumnNames=False, commentLines=[0])
            nModes = int(f.header[0].split()[-1])
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise WrongFormatError(
                'Unable to read the HAWCStab2 Campbell header'
            ) from error
        if nModes <= 0:
            raise BrokenFormatError('The Campbell file reports no modes')
        nCols = f.data.shape[1]-1
        colsF = ['F{}'.format(i+1) for i in range(nModes)]
        colsD = ['D{}'.format(i+1) for i in range(nModes)]
        cols=colsF.copy()
        if nCols/nModes in [2,3]:
            cols+=colsD
        if nCols/nModes in [3]:
            colsR = ['R{}'.format(i+1) for i in range(nModes)]
            cols+=colsR
        else:
            colsR=[]
        if len(cols) != nCols:
            raise BrokenFormatError(
                'Campbell data columns are inconsistent with the number of modes'
            )
        f.data.columns = ['Wind_[m/s]'] + cols
        self.data = f.toDataFrame()

    def _toDataFrame(self):
        return self.data
