import numpy as np
import pandas as pd
import os
import re
try:
    from .file import File, WrongFormatError, BrokenFormatError
except:
    File = dict
    class WrongFormatError(Exception): pass
    class BrokenFormatError(Exception): pass


class HAWCStab2IndFile(File):

    @staticmethod
    def defaultExtensions():
        return ['.ind', '.txt']

    @staticmethod
    def formatName():
        return 'HAWCStab2 induction file'

    def _read(self, *args, **kwargs):
        # Reading header line
        with open(self.filename,'r',encoding=self.encoding) as f:
            header = f.readline().strip()
        if len(header)<=0 or header[0]!='#':
            raise WrongFormatError(
                'Ind File {}: header line does not start with `#`.'.format(
                    self.filename
                )
            )
        # Extracting column names
        header       = '00'+header[1:].strip()
        num_and_cols = [s.strip()+']' for s in header.split(']')[:-1]]
        cols         = [col[2:].strip().replace(' ','_')  for col in num_and_cols]
        cols         = [col.replace('[','_[').replace('__','_')  for col in cols]
        # Determining type based on number of columns (NOTE: could use col names as well maybe)
        NumCol2Type = {38: 'ind', 14: 'fext', 18: 'defl'}
        try:
            self.type = NumCol2Type[len(cols)]
        except Exception as e:    
            raise WrongFormatError('Ind File {}: '.format(self.filename))
        self.colNames=cols

        # Reading numerical data
        try:
            self.data = np.atleast_2d(np.loadtxt(self.filename, skiprows=1))
        except Exception as e:    
            raise BrokenFormatError('Ind File {}: '.format(self.filename)+e.args[0])

        if self.data.shape[1]!=len(cols):
            raise BrokenFormatError(
                'Ind File {}: inconsistent number of header columns and data '
                'columns.'.format(self.filename)
            )

        # Extracting wind speed from filename 
        match = re.search(
            r'(?:^|_)u?(\d+(?:\.\d+)?)\.ind$',
            os.path.basename(self.filename),
            flags=re.IGNORECASE,
        )
        self.wsp = float(match.group(1)) / 1000 if match else np.nan

    def _toDataFrame(self):
        key = '{:s} - ws={:06.3f}'.format(self.type,self.wsp)
        df= pd.DataFrame(data=self.data, columns=self.colNames)
        df.columns.name=key
        return df
        #dfs = {key: pd.read_csv(self.filename, delim_whitespace=True, names=cols, skiprows=1)}
        #return dfs
