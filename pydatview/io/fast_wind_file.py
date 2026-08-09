import os
import re
import struct

import numpy as np
import pandas as pd

from .csv_file import CSVFile
from .file import isBinary, WrongFormatError, BrokenFormatError

class FASTWndFile(CSVFile):

    @staticmethod
    def defaultExtensions():
        return ['.wnd']

    @staticmethod
    def formatName():
        # Keep the historical name because scan preferences use it as a key.
        return 'FAST determ. wind file'

    def __init__(self, *args, **kwargs):
        self.colNames=['Time','WindSpeed','WindDir','VertSpeed','HorizShear','VertShear','LinVShear','GustSpeed']
        self.units=['[s]','[m/s]','[deg]','[m/s]','[-]','[-]','[-]','[m/s]']
        self._formatted_columns=['{}_{}'.format(c,u) for c,u in zip(self.colNames,self.units)]
        self._binary_wind = None

        header=[]
        header+=['!Wind file.']
        header+=['!Time  Wind     Wind	Vert.       Horiz.      Vert.       LinV        Gust']
        header+=['!      Speed    Dir    Speed       Shear		Shear       Shear       Speed']

        super(FASTWndFile, self).__init__(
            sep=' ',
            commentChar='!',
            colNames=self._formatted_columns,
            header=header,
            *args,
            **kwargs
        )

    def _read(self, *args, **kwargs):
        if isBinary(self.filename):
            self._binary_wind = self._read_bladed_binary()
            self.data = []
            return

        # InflowWind permits any number of leading comment lines containing
        # !, #, or %, and permits commas as well as whitespace as separators.
        # CSVFile only supports one leading comment character and one fixed
        # separator, so parse this small, well-defined format directly.
        self._binary_wind = None
        header = []
        data_start = None
        with open(self.filename, 'r', encoding=self.encoding, errors='surrogateescape') as stream:
            for line_number, line in enumerate(stream):
                stripped = line.strip()
                if not stripped or any(marker in line for marker in ('!', '#', '%')):
                    header.append(stripped)
                    continue
                data_start = line_number
                break
        if data_start is None:
            raise WrongFormatError('No numeric wind data found in {}'.format(self.filename))
        try:
            data = pd.read_csv(
                self.filename,
                sep=r'[\s,]+',
                engine='python',
                skiprows=data_start,
                header=None,
            )
        except (ValueError, pd.errors.ParserError) as exc:
            raise WrongFormatError('Invalid uniform wind data: {}'.format(exc))
        if data.empty or data.shape[1] < 2 or data.shape[1] > len(self._formatted_columns):
            raise WrongFormatError(
                'Uniform wind files must contain between 2 and 8 numeric columns; found {}'.format(
                    data.shape[1]
                )
            )
        try:
            data = data.apply(pd.to_numeric, errors='raise')
        except (TypeError, ValueError) as exc:
            raise WrongFormatError('Uniform wind data contains a non-numeric value: {}'.format(exc))
        data.columns = self._formatted_columns[:data.shape[1]]
        self.data = data
        self.header = header
        self.commentLines = list(range(data_start))

    @staticmethod
    def _read_exact(stream, fmt, label):
        size = struct.calcsize(fmt)
        raw = stream.read(size)
        if len(raw) != size:
            raise BrokenFormatError('Unexpected end of binary .wnd while reading {}'.format(label))
        return struct.unpack(fmt, raw)

    @staticmethod
    def _summary_path(filename):
        root = os.path.splitext(filename)[0]
        direct = root + '.sum'
        if os.path.isfile(direct):
            return direct
        directory = os.path.dirname(filename) or '.'
        target = os.path.basename(root + '.sum').lower()
        try:
            for candidate in os.listdir(directory):
                if candidate.lower() == target:
                    return os.path.join(directory, candidate)
        except OSError:
            pass
        return None

    @staticmethod
    def _read_bladed_summary(filename):
        values = {
            'clockwise': False,
            'left_hand_rule': False,
            'periodic': False,
            'height_offset': 0.0,
            'ref_height': None,
            'mean_wind_speed': None,
            'ti': [None, None, None],
        }
        summary = FASTWndFile._summary_path(filename)
        if summary is None:
            return values

        number = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?'
        with open(summary, 'r', encoding='utf-8', errors='ignore') as stream:
            for line in stream:
                upper = line.upper()
                match = re.search(number, line)
                value = float(match.group(0).replace('D', 'E').replace('d', 'e')) if match else None
                if 'CLOCKWISE' in upper:
                    first = upper.strip()[:1]
                    values['clockwise'] = first in ('T', 'Y')
                elif 'HUB HEIGHT' in upper or 'ZHUB' in upper:
                    values['ref_height'] = value
                elif 'UBAR' in upper:
                    values['mean_wind_speed'] = value
                elif 'TI(U)' in upper:
                    values['ti'][0] = value
                elif 'TI(V)' in upper:
                    values['ti'][1] = value
                elif 'TI(W)' in upper:
                    values['ti'][2] = value
                elif 'HEIGHT OFFSET' in upper:
                    values['height_offset'] = value or 0.0
                if 'PERIODIC' in upper:
                    values['periodic'] = True
                if 'BLADED LEFT-HAND RULE' in upper:
                    values['left_hand_rule'] = True
        return values

    def _read_bladed_binary(self):
        """Read the TurbSim/Bladed full-field binary variant of ``.wnd``."""
        summary = self._read_bladed_summary(self.filename)
        with open(self.filename, 'rb') as stream:
            file_id, = self._read_exact(stream, '<h', 'file identifier')
            header_ti = [None, None, None]
            ref_height = summary['ref_height']

            if file_id == -99:
                turbulence_type, = self._read_exact(stream, '<h', 'turbulence type')
                if turbulence_type in (1, 2):
                    n_components = 1
                elif turbulence_type in (3, 5):
                    n_components = 3
                elif turbulence_type == 4:
                    fields = self._read_exact(stream, '<i6f', 'improved von Karman header')
                    n_components = fields[0]
                    ref_height = fields[3]
                    header_ti = list(fields[4:7])
                elif turbulence_type in (7, 8):
                    _header_size, n_components = self._read_exact(
                        stream, '<2i', 'general turbulence header'
                    )
                else:
                    raise BrokenFormatError(
                        'Unsupported Bladed .wnd turbulence type {}'.format(turbulence_type)
                    )

                sub = self._read_exact(stream, '<3fi4f4i', 'grid header')
                dz, dy, dx = sub[:3]
                nominal_steps = 2 * sub[3]
                mean_wind_speed = sub[4]
                nz, ny = sub[-2:]
                if n_components == 3:
                    self._read_exact(stream, '<6f', 'component length scales')
                if turbulence_type == 7:
                    self._read_exact(stream, '<2f', 'Kaimal header')
                elif turbulence_type == 8:
                    self._read_exact(stream, '<6f3i2f3i2f', 'Mann header')
            elif file_id in (-1, -2, -3):
                # The first int16 is also the old-format component identifier.
                rest = self._read_exact(stream, '<12h', 'legacy Bladed header')
                values = (file_id,) + rest
                n_components = -values[0]
                dz, dy, dx = (0.001 * values[i] for i in (1, 2, 3))
                nominal_steps = 2 * values[4]
                mean_wind_speed = 0.1 * values[5]
                nz, ny = values[11] // 1000, values[12] // 1000
                if n_components == 3:
                    self._read_exact(stream, '<6h', 'legacy component header')
            else:
                raise BrokenFormatError(
                    'Binary .wnd file identifier {} is not a supported Bladed format'.format(file_id)
                )

            raw_bytes = stream.read()

        if n_components not in (1, 2, 3) or ny <= 0 or nz <= 0:
            raise BrokenFormatError(
                'Invalid Bladed .wnd dimensions: {} components, ny={}, nz={}'.format(
                    n_components, ny, nz
                )
            )
        if not all(np.isfinite(value) and value > 0 for value in (dx, dy, dz)):
            raise BrokenFormatError(
                'The binary .wnd has invalid grid spacing: dx={}, dy={}, dz={}'.format(
                    dx, dy, dz
                )
            )
        values_per_step = n_components * ny * nz
        raw = np.frombuffer(raw_bytes, dtype='<i2')
        if raw.size == 0 or raw.size % values_per_step:
            raise BrokenFormatError(
                'Binary .wnd data size is inconsistent with its {}x{} grid'.format(ny, nz)
            )
        nt = raw.size // values_per_step
        if nominal_steps > 0 and nt not in (nominal_steps, nominal_steps + 1):
            raise BrokenFormatError(
                'Binary .wnd header declares {} time steps but contains {}'.format(
                    nominal_steps, nt
                )
            )

        summary_ti = summary['ti']
        ti = [
            header_ti[i] if header_ti[i] is not None and header_ti[i] > 0 else summary_ti[i]
            for i in range(3)
        ]
        if any(ti[i] is None or ti[i] <= 0 for i in range(n_components)):
            raise BrokenFormatError(
                'The binary .wnd needs positive turbulence intensities in its header or companion .sum file'
            )
        if summary['mean_wind_speed'] is not None:
            mean_wind_speed = summary['mean_wind_speed']
        if not np.isfinite(mean_wind_speed) or mean_wind_speed <= 0:
            raise BrokenFormatError('The binary .wnd has an invalid mean wind speed')

        field = raw.reshape((n_components, ny, nz, nt), order='F').astype(np.float32)
        if summary['clockwise']:
            field = field[:, ::-1, :, :]
        scale = 0.001 * mean_wind_speed * np.asarray(ti[:n_components]) / 100.0
        if summary['left_hand_rule'] and n_components > 1:
            scale[1] *= -1.0
        field *= scale[:, None, None, None]
        field[0] += mean_wind_speed
        field = field.transpose(0, 3, 1, 2)
        if n_components < 3:
            missing = np.zeros((3 - n_components, nt, ny, nz), dtype=field.dtype)
            field = np.concatenate((field, missing))

        from .turbsim_file import TurbSimFile
        wind = TurbSimFile()
        wind.filename = self.filename
        wind['u'] = field
        wind['uTwr'] = np.zeros((3, nt, 0), dtype=field.dtype)
        wind['info'] = 'Bladed-style full-field wind read from {}'.format(os.path.basename(self.filename))
        wind['ID'] = file_id
        wind['dt'] = dx / mean_wind_speed
        wind['y'] = np.arange(ny) * dy
        wind['y'] -= np.mean(wind['y'])
        center_height = (
            ref_height if ref_height is not None else 0.5 * dz * (nz - 1)
        ) - summary['height_offset']
        wind['z'] = np.arange(nz) * dz + center_height - 0.5 * dz * (nz - 1)
        wind['t'] = np.arange(nt) * wind['dt']
        wind['zTwr'] = np.asarray([], dtype=float)
        wind['zRef'] = ref_height if ref_height is not None else center_height
        wind['uRef'] = mean_wind_speed
        wind['periodic'] = summary['periodic']
        return wind

    def _write(self, *args, **kwargs):
        super(FASTWndFile, self)._write(*args, **kwargs)


    def _toDataFrame(self):
        if self._binary_wind is not None:
            return self._binary_wind.toDataFrame()
        return self.data

    def to2DFields(self, **kwargs):
        if self._binary_wind is not None:
            return self._binary_wind.to2DFields(**kwargs)
        return super(FASTWndFile, self).to2DFields(**kwargs)


# --------------------------------------------------------------------------------}
# --- Functions specific to file type  
# --------------------------------------------------------------------------------{
    def stepWind(self,WSstep=1,WSmin=3,WSmax=25,tstep=100,dt=0.5,tmin=0,tmax=999):
        """ Set the wind file to a step wind 
        tstep: can be an array of size 2 [tstepmax tstepmin]

        
        """
            
        Steps= np.arange(WSmin,WSmax+WSstep,WSstep)
        if hasattr(tstep,'__len__'):
            tstep = np.around(np.linspace(tstep[0], tstep[1], len(Steps)),0)
        else:
            tstep = len(Steps)*[tstep]
        nCol = len(self.colNames)
        nRow = len(Steps)*2
        M = np.zeros((nRow,nCol));
        M[0,0] = tmin
        M[0,1] = WSmin
        for i,s in enumerate(Steps[:-1]):
            M[2*i+1,0] = tmin + tstep[i]-dt 
            M[2*i+2,0] = tmin + tstep[i]
            tmin +=tstep[i]
            M[2*i+1,1] = Steps[i]
            if i<len(Steps)-1:
                M[2*i+2,1] = Steps[i+1]
            else:
                M[2*i+2,1] = Steps[-1]
        M[-1,0]= max(tmax, tmin+tstep[-1])
        M[-1,1]= WSmax
        self.data=pd.DataFrame(data=M,columns=self.colNames)
