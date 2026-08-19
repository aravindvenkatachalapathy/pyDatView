import os

import pandas as pd

from .file import File


class ExcelFile(File):

    @staticmethod
    def defaultExtensions():
        return ['.xls', '.xlsx']

    @staticmethod
    def formatName():
        return 'Excel file'

    def _read(self, **kwargs):
        self.data = {}
        default_engine = (
            'xlrd' if os.path.splitext(self.filename)[1].lower() == '.xls'
            else 'openpyxl'
        )
        engine = kwargs.pop('engine', default_engine)

        with pd.ExcelFile(self.filename, engine=engine, **kwargs) as workbook:
            for sheet_name in workbook.sheet_names:
                frame = workbook.parse(sheet_name, header=None)
                frame.dropna(how='all', axis=0, inplace=True)
                frame.dropna(how='all', axis=1, inplace=True)
                if frame.shape[0] > 0:
                    frame = (
                        frame.rename(columns=frame.iloc[0])
                        .drop(frame.index[0])
                        .reset_index(drop=True)
                    )
                    self.data[sheet_name] = frame

    def _write(self, **kwargs):
        extension = os.path.splitext(self.filename)[1].lower()
        if extension == '.xls':
            raise NotImplementedError(
                'Writing legacy .xls files is not supported; use .xlsx instead'
            )

        with pd.ExcelWriter(
            self.filename, engine='openpyxl', **kwargs
        ) as writer:
            for sheet_name, frame in self.data.items():
                frame.to_excel(writer, sheet_name=sheet_name, index=False)

    def __repr__(self):
        return 'Class ExcelFile (attributes: data)\n'

    def _toDataFrame(self):
        if len(self.data) == 1:
            return self.data[next(iter(self.data))]
        return self.data
