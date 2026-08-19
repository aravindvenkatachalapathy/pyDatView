"""Reader and writer for general MATLAB MAT files."""

import numpy as np
import pandas as pd
import scipy.io

from .file import BrokenFormatError, File


class MatlabMatFile(File):

    @staticmethod
    def defaultExtensions():
        return ['.mat']

    @staticmethod
    def formatName():
        return 'MATLAB MAT file'

    def _read(self, **kwargs):
        kwargs.setdefault('squeeze_me', True)
        try:
            contents = scipy.io.loadmat(self.filename, **kwargs)
        except NotImplementedError as error:
            raise BrokenFormatError(
                'MATLAB v7.3/HDF5 files are not supported by scipy.io.loadmat'
            ) from error
        except (OSError, ValueError, scipy.io.matlab.MatReadError) as error:
            raise BrokenFormatError(
                'scipy could not read the MATLAB file: {}'.format(error)
            ) from error

        self.data = {
            name: value
            for name, value in contents.items()
            if not name.startswith('__')
        }
        if not self.data:
            raise BrokenFormatError('The MATLAB file contains no data variables')

    def _write(self, **kwargs):
        scipy.io.savemat(self.filename, self.data, **kwargs)

    @staticmethod
    def _variable_to_frame(name, value):
        array = np.asarray(value)
        if array.dtype.names is not None:
            return None
        if array.dtype == object and any(
            np.asarray(item).ndim > 0 for item in array.reshape(-1)
        ):
            return None

        if array.ndim == 0:
            return pd.DataFrame({name: [array.item()]})
        if array.ndim == 1 or 1 in array.shape:
            return pd.DataFrame({name: array.reshape(-1)})
        if array.ndim == 2:
            return pd.DataFrame(
                array,
                columns=['{}_{}'.format(name, i) for i in range(array.shape[1])],
            )

        indices = np.indices(array.shape).reshape(array.ndim, -1).T
        frame = pd.DataFrame(
            indices,
            columns=['dim_{}'.format(i) for i in range(array.ndim)],
        )
        frame[name] = array.reshape(-1)
        return frame

    def _toDataFrame(self):
        frames = {}
        for name, value in self.data.items():
            frame = self._variable_to_frame(name, value)
            if frame is not None:
                frames[name] = frame

        if len(frames) == 1:
            return frames[next(iter(frames))]
        return frames

    def __repr__(self):
        return '<MatlabMatFile object with variables: {}>'.format(
            ', '.join(self.data)
        )
