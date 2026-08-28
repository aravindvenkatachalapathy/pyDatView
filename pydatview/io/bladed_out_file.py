import os
import numpy as np
import re
import pandas as pd
import glob
import shlex
try:
    import pydatview_fastio
except ImportError:
    pydatview_fastio = None
try:
    from .file import BrokenFormatError, EmptyFileError, File, WrongFormatError
except:
    File = dict
    class EmptyFileError(Exception): pass
    class WrongFormatError(Exception): pass
    class BrokenFormatError(Exception): pass


class _BladedLazyMatrix:
    """Column-oriented memory-mapped view of a binary Bladed dataset."""

    ndim = 2

    def __init__(self, filename, sensor_info):
        self.filename = filename
        self.dtype = np.dtype(sensor_info['Precision'])
        self.n_sections = int(sensor_info['nSections'])
        self.n_sensors = int(sensor_info['nSensors'])
        self.n_major = int(sensor_info['nMajor'])
        values_per_step = self.n_sections * self.n_sensors
        if self.n_major == 0:
            value_count = os.path.getsize(filename) // self.dtype.itemsize
            self.n_major = value_count // values_per_step
        self.shape = (self.n_major, values_per_step)
        self._memmap = None

    def _values(self):
        if self._memmap is None:
            self._memmap = np.memmap(
                self.filename,
                dtype=self.dtype,
                mode='r',
                shape=(self.n_major, self.n_sections, self.n_sensors),
                order='C',
            )
        return self._memmap

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise IndexError('Bladed lazy data requires row and column indices')
        row_selector, column_selector = key
        if not np.isscalar(column_selector):
            raise IndexError('Bladed lazy data is loaded one channel at a time')
        column_index = int(column_selector)
        if column_index < 0 or column_index >= self.shape[1]:
            raise IndexError('Bladed channel index is outside the dataset')
        section_index, sensor_index = divmod(
            column_index,
            self.n_sensors,
        )
        return self._values()[row_selector, section_index, sensor_index]

        
# --------------------------------------------------------------------------------}
# --- Helper functions 
# --------------------------------------------------------------------------------{
def read_bladed_sensor_file(sensorfile):        
    """ 
    Extract relevant informations from a bladed sensor file
    """
    with open(sensorfile, 'r') as fid:
        sensorLines = fid.readlines()

    dat=dict() # relevant info in sensor file
    
    ## read sensor file line by line (just read up to line 20)
    #while i < 17: 
    for i, t_line in enumerate(sensorLines):
        if i>30:
            break
        t_line = t_line.replace('\t',' ')
        
        if t_line.startswith('NDIMENS'):
            # check what is matrix dimension of the file. For blade & tower,  
            # the matrix is 3-dimensional. 
            temp =  t_line[7:].strip().split()
            dat['NDIMENS'] = int(temp[-1]);

        elif t_line.startswith('DIMENS'):
            # check what is the size of the matrix
            # for example, it can be 11x52500 or 12x4x52500            
            temp =  t_line[6:].strip().split()
            dat['nSensors'] = int(temp[0])
            dat['nMajor']  = int(temp[dat['NDIMENS']-1])
            if dat['NDIMENS'] == 2:
                dat['nSections'] = 1
                dat['SectionList'] = []

        elif t_line.startswith('FORMAT'):
            # precision: n/a, R*4, R*8, I*4 
            temp = t_line[7:].strip()
            dat['Precision'] = np.float32
            if temp[-1] == '8':
                dat['Precision'] = np.float64

        elif t_line.startswith('GENLAB'):
            # category of the file you are reading:
            dat['category'] = t_line[6:].strip().replace('\'','')

        elif t_line.startswith('AXIVAL'):
            # Section on the 3rd dimension you are reading
            # sometimes, the info is written on "AXITICK"
            temp = t_line[7:].split()
            dat['SectionList'] = np.array(temp, dtype=float)
            dat['nSections'] = len(dat['SectionList'])

        elif t_line.startswith('AXITICK'):
            # Section on the 3rd dimension you are reading
            # sometimes, the info is written on "AXIVAL"
            # Check next line, we concatenate if doesnt start with AXISLAB (Might need more cases)
            try:
                # Combine the strings into one string
                combined_string = ''.join(sensorLines)
                # Search for a regex pattern that spans across multiple strings
                line = re.search(r'(?<=AXITICK).+?(?=(AXISLAB|NVARS))', combined_string, flags=re.DOTALL)
                line=line.group(0)
                # Replace consecutive whitespace characters with a single space
                t_line = re.sub(r'\s+', ' ', line)
            except:
                pass

            temp = t_line.strip()
            temp = temp.strip('\'').split('\' \'')
            dat['SectionList'] = np.array(temp, dtype=str)
            dat['nSections'] = len(dat['SectionList'])

        elif t_line.startswith('VARIAB'):
            # channel names, NOTE: either quoted, non-quoted, and a mix of both
            # Check next line, we concatenate if doesnt start with AXISLAB (Might need more cases)
            try:
                nextLine=sensorLines[i+1].strip()
                if not nextLine.startswith('VARUNIT'):
                    t_line = t_line.strip()+' '+nextLine
            except:
                pass
            dat['ChannelName'] = shlex.split(t_line[6:])

        elif t_line.startswith('VARUNIT'):
            # channel units:         
            # Check next line, we concatenate if doesnt start with AXISLAB (Might need more cases)
            try:
                nextLine=sensorLines[i+1].strip()
                if not nextLine.startswith('AXISLAB'):
                    t_line = t_line.strip()+' '+nextLine
            except:
                pass
            def repUnits(s):
                s = s.replace('[[','[').replace(']]',']')
                s = s.replace('TT','s^2').replace('T','s').replace('A','rad')
                s = s.replace('P','W').replace('L','m').replace('F','N').replace('M','kg')
                return s
            dat['ChannelUnit']=[repUnits(s) for s in shlex.split(t_line[7:].strip())]

        elif t_line.startswith('MIN '):
            dat['MIN'] = float(t_line[3:].strip()) # Start time?

        elif t_line.startswith('STEP'):
            dat['STEP'] = float(t_line[4:].strip()) # DT?


    NeededKeys=['ChannelName','nSensors','nMajor','nSections']
    if not all(key in dat.keys() for key in NeededKeys):
        raise BrokenFormatError('Broken or unsupported format. Some necessary keys where not found in the bladed sensor file: {}'.format(sensorfile))

    if len(dat['ChannelName']) != dat['nSensors']:
        raise BrokenFormatError('Broken or unsupported format. Wrong number of channels while reading bladed sensor file: {}'.format(sensorfile))
        # if number of channel names are not matching with Sensor number then create dummy ones:
        #dat['ChannelName'] = ['Channel' + str(ss) for ss in range(dat['nSensors'])]

    return dat


def organize_bladed_3d_columns(**info):
    """Return flattened Bladed 3D channel names and units."""
    names = []
    units = []
    for sec in info['SectionList']:
        for chan, unit in zip(info['ChannelName'], info['ChannelUnit']):
            try:
                names.append(str(np.around(float(sec), 2)) + 'm-' + chan)
            except ValueError:
                names.append(str(sec) + '-' + chan)
            units.append(unit)
    return names, units


def OrgData(data, **info):
    """ Flatten 3D field into 2D table"""
    # since some of the matrices are 3 dimensional, we want to make all 
    # to 2d matrix, so I am organizing them here:
    if info['NDIMENS'] == 3:
        data = data.reshape(info['nMajor'], -1)
        SName, SUnit = organize_bladed_3d_columns(**info)
        info['ChannelName'] = SName
        info['ChannelUnit'] = SUnit
    else:
        pass # Nothing to do for 2D

    return data, info



def read_bladed_output(sensorFilename, readTimeFilesOnly=False):
    """
    read a bladed sensor file and data file, reorganize a 3D file into 2D table
    """
    # --- Read sensor file and extract relevant informations
    sensorInfo = read_bladed_sensor_file(sensorFilename)
    nSensors   = sensorInfo['nSensors']
    nMajor     = sensorInfo['nMajor']
    nSections  = sensorInfo['nSections']
    hasTime = 'MIN' in sensorInfo and 'STEP' in sensorInfo
    # --- Return if caller only wants time series
    if readTimeFilesOnly and not hasTime:
        return [], {}
    
    # --- Read data file
    dataFilename = sensorFilename.replace('%','$')

    if isBinary(dataFilename):            # it is binary            

        if pydatview_fastio is not None:
            try:
                data, nMajorRust = pydatview_fastio.read_bladed_binary(
                    dataFilename,
                    nMajor,
                    nSections,
                    nSensors,
                    sensorInfo['NDIMENS'],
                    sensorInfo['Precision'] == np.float64,
                )
                data = np.asarray(data)
                sensorInfo['nMajor'] = nMajorRust
                sensorInfo['loader_backend'] = 'rust'
                if sensorInfo['NDIMENS'] == 3:
                    sensorInfo['ChannelName'], sensorInfo['ChannelUnit'] = organize_bladed_3d_columns(**sensorInfo)
                print('[pyDatView] Bladed binary load: Rust ({})'.format(dataFilename))
                return data, sensorInfo
            except Exception as e:
                print('[pyDatView] Bladed binary Rust load failed, falling back to Python/NumPy ({}): {}'.format(dataFilename, e))
        print('[pyDatView] Bladed binary load: Python/NumPy ({})'.format(dataFilename))

        with open(os.path.join(dataFilename), 'rb') as fid_2:
            data = np.fromfile(fid_2, sensorInfo['Precision'])

        try:
            if nMajor==0:
                nMajor=int(np.floor(len(data)/nSections/nSensors))
                data=data[0:nMajor*nSections*nSensors]
                sensorInfo['nMajor']=nMajor
            if sensorInfo['NDIMENS'] == 3:
                data = np.reshape(data,(nMajor, nSections, nSensors), order='C')

            elif sensorInfo['NDIMENS'] == 2:
                data = np.reshape(data,(nMajor,nSensors), order='C')
        except:
            print('>>> Failed to reshape binary file {}'.format(dataFilename))
            raise

            
    else:
        #print('it is ascii', NDIMENS)
        if sensorInfo['NDIMENS'] == 2:
            try:
                # Data is stored as time, signal, we reshape to signal, time
                data = np.loadtxt(dataFilename)
            except ValueError as e:
                # Most likely this was a binary file...
                data = np.empty((nMajor, nSensors)) * np.nan
                print('>>> Value error while reading 2d ascii file: {}'.format(dataFilename))
                raise e
            except:
                data = np.empty((nMajor, nSensors)) * np.nan
                print('>>> Failed to read 2d ascii file: {}'.format(dataFilename))
                raise

       
        elif sensorInfo['NDIMENS'] == 3:
            try:
                # Data is stored as sections, time, signal, we reshape to signal, section, time
                data = np.loadtxt(dataFilename).reshape((nMajor, nSections, nSensors),order='C')
            except:
                data = np.empty((nMajor, nSections, nSensors)) * np.nan
                print('>>> Failed to read 3d ascii file: {}'.format(dataFilename))

    return OrgData(data, **sensorInfo)


class BladedFile(File):
    r"""
    Read a Bladed out put file (current version is only binary files)
    
    Main methods:
        read: it finds all % and $ files based on selected .$PJ file and calls "DataValue" to read data from all those files
        toDataFrame: create Pandas dataframe output
        
    Main data stored:
         self.dataSets: dictionary of datasets, for each "length" of data

    example: 
        filename = r'h:\004_Loads\Sim\Bladed\003\Ramp_up\Bladed_out_ascii.$04'        
        f = BladedFile(filename)
        print(f.dataSets.keys())
        df = f.toDataFrame()
        
    """ 
    @staticmethod
    def defaultExtensions():
        return ['.%*', '.$*'] 

    @staticmethod
    def formatName():
        return 'Bladed output file'
    
    def __init__(self, filename=None, **kwargs):
        self.filename = filename
        if filename:
            self.read(**kwargs)
        
    def read(self, filename=None, **kwargs):
        """ read self, or read filename if provided """
        if filename:
            self.filename = filename
        if not self.filename:
            raise Exception('No filename provided')
        if not os.path.isfile(self.filename):
            raise OSError(2,'File not found:',self.filename)
        if os.stat(self.filename).st_size == 0:
            raise EmptyFileError('File is empty:',self.filename)
        # Calling children function
        self._read(**kwargs)
    
    def _read(self):
        """ 
        Read a bladed output file, data are in *.$II and sensors in *%II. 
         - If the file is a *$PJ file, all output files are read
         - Otherwise only the current file is read 
        """

        basename, ext = os.path.splitext(self.filename)
        is_project = ext.lower()=='.$pj'
        if is_project:
            readTimeFilesOnly=True
            searchPattern = basename + '.%[0-9][0-9]*' # find all files in the folder
        else:
            readTimeFilesOnly=False
            searchPattern = basename + ext.replace('$','%') # sensor file name
        
        # Look for files matching pattern
        files = glob.glob(searchPattern)

        # We'll store the data in "dataSets",dictionaries
        dataSets={}

        if len(files)==0:
            e= FileNotFoundError(searchPattern)
            e.filename=(searchPattern)
            raise e
        elif len(files)==1:
            readTimeFilesOnly=False

        files.sort()

        for i,filename in enumerate(files):

            dataFilename = filename.replace('%','$')
            if is_project:
                if not os.path.isfile(dataFilename):
                    print('>>> Missing datafile: {}'.format(dataFilename))
                    continue
                info = read_bladed_sensor_file(filename)
                has_time = 'MIN' in info and 'STEP' in info
                if readTimeFilesOnly and not has_time:
                    continue
                if isBinary(dataFilename):
                    matrix = _BladedLazyMatrix(dataFilename, info)
                    info['nMajor'] = matrix.n_major
                    if info['NDIMENS'] == 3:
                        info['ChannelName'], info['ChannelUnit'] = (
                            organize_bladed_3d_columns(**info)
                        )
                    category = (
                        info.get('category')
                        or os.path.splitext(os.path.basename(filename))[1]
                    )
                    dataset_name = category
                    if dataset_name in dataSets:
                        dataset_name = '{} ({})'.format(
                            category,
                            os.path.splitext(filename)[1].lstrip('.%$'),
                        )
                    sensors = list(info['ChannelName'])
                    units = list(info['ChannelUnit'])
                    dset = {
                        'data': None,
                        'sensors': sensors,
                        'units': units,
                        'name': dataset_name,
                        '_lazy_plot_data': matrix,
                        '_n_major': matrix.n_major,
                    }
                    if has_time:
                        dset['_time'] = (
                            np.arange(matrix.n_major, dtype=np.float64)
                            * info['STEP']
                            + info['MIN']
                        )
                        dset['sensors'].insert(0, 'Time')
                        dset['units'].insert(0, 's')
                        dset['_numpy_plot_column_offset'] = 2
                    else:
                        dset['_numpy_plot_column_offset'] = 1
                    dataSets[dataset_name] = dset
                    continue
            try:
                # Call "Read_bladed_file" function to Read and store data:
                data, info = read_bladed_output(filename, readTimeFilesOnly=readTimeFilesOnly)    
            except FileNotFoundError as e:
                print('>>> Missing datafile: {}'.format(e.filename))
                if len(files)==1:
                    raise e
                continue
            except ValueError as e:
                print('>>> ValueError while reading: {}'.format(dataFilename))
                if len(files)==1:
                    raise e
                continue
            except:
                raise 
                print('>>> Misc error while reading: {}'.format(dataFilename))
                if len(files)==1:
                    raise 
                continue
            if len(data)==0:
                print('>>> Skipping file since no time present {}'.format(filename))
                continue
            
            # we use number of data as key, but we'll use "name" later
            key = info['nMajor']

            if is_project:
                dset = {}
                native_plot_data = data if info.get('loader_backend') == 'rust' else None
                has_time = 'MIN' in info and 'STEP' in info
                if has_time:
                    time = np.arange(info['nMajor'])*info['STEP'] + info['MIN']
                    dset['_time'] = time
                    info['ChannelName'].insert(0, 'Time')
                    info['ChannelUnit'].insert(0, 's')

                category = info.get('category') or os.path.splitext(os.path.basename(filename))[1]
                dataset_name = category
                if dataset_name in dataSets:
                    dataset_name = '{} ({})'.format(category, os.path.splitext(filename)[1].lstrip('.%$'))
                dset['data'] = data
                dset['sensors'] = info['ChannelName']
                dset['units'] = info['ChannelUnit']
                dset['name'] = dataset_name
                if native_plot_data is not None:
                    dset['_numpy_plot_data'] = native_plot_data
                    dset['_numpy_plot_column_offset'] = 2 if has_time else 1
                dataSets[dataset_name] = dset
                continue
            
            if key in dataSets.keys():
                # dataset with this length are already present, we concatenate
                dset = dataSets[key]
                dset['data'] =  np.column_stack((dset['data'], data))
                dset.pop('_numpy_plot_data', None)
                dset.pop('_numpy_plot_column_offset', None)
                dset['sensors'] += info['ChannelName']
                dset['units']   += info['ChannelUnit']
                dset['name']  = 'Misc_'+str(key)

            else:
                # We add a new dataset for this length
                dataSets[key] = {}
                dset = dataSets[key]
                native_plot_data = data if info.get('loader_backend') == 'rust' else None
                # We force a time vector when possible
                has_time = 'MIN' in info and 'STEP' in info
                if has_time:
                    time = np.arange(info['nMajor'])*info['STEP'] + info['MIN']
                    dset['_time'] = time
                    info['ChannelName'].insert(0, 'Time')
                    info['ChannelUnit'].insert(0, 's')

                dset['data']    = data
                dset['sensors'] = info['ChannelName']
                dset['units']   = info['ChannelUnit']
                dset['name']    = info['category']
                if native_plot_data is not None:
                    dset['_numpy_plot_data'] = native_plot_data
                    dset['_numpy_plot_column_offset'] = 2 if has_time else 1

        if is_project:
            self.dataSets = dataSets
        else:
            # Check if we have "many" misc, if only one, replace by "Misc"
            keyMisc = [k for k,v in dataSets.items() if v['name'].startswith('Misc_')]
            if len(keyMisc)==1:
                #dataSets[keyMisc[0]]['name']='Misc'
                # We keep only one dataset for simplicity
                self.dataSets= {'Misc': dataSets[keyMisc[0]]}
            else:
                # Instead of using nMajor as key, we use the "name"
                self.dataSets= {v['name']: v for (k, v) in dataSets.items()}

                
    def toDataFrame(self):        
        dfs={}
        for k,dset in self.dataSets.items():
            BL_ChannelUnit = [ name+' ['+unit+']' for name,unit in zip(dset['sensors'],dset['units'])]
            if '_lazy_plot_data' in dset:
                placeholder = pd.Series(
                    index=pd.RangeIndex(dset['_n_major']),
                    dtype=pd.SparseDtype(dset['_lazy_plot_data'].dtype, np.nan),
                ).array
                start = 1 if '_time' in dset else 0
                df = pd.DataFrame({
                    column: placeholder
                    for column in BL_ChannelUnit[start:]
                })
                if '_time' in dset:
                    df.insert(0, BL_ChannelUnit[0], dset['_time'])
                df.attrs['pydatview'] = {
                    'lazy_values': True,
                    'lazy_column_offset': dset[
                        '_numpy_plot_column_offset'
                    ],
                    'source_variable': str(k),
                }
            elif '_time' in dset:
                df = pd.DataFrame(data=dset['data'], columns=BL_ChannelUnit[1:])
                df.insert(0, BL_ChannelUnit[0], dset['_time'])
            else:
                df = pd.DataFrame(data=dset['data'], columns=BL_ChannelUnit)
            # remove duplicate columns
            if df.columns.duplicated().any():
                df = df.loc[:,~df.columns.duplicated()]
            df.columns.name = k # hack for pyDatView when one dataframe is returned
            dfs[k] = df
        if len(dfs)==1:
            return dfs[next(iter(dfs))]
        else:
            return dfs

    def get_numpy_plot_data(self, table_name=''):
        dset = self.dataSets.get(table_name)
        if dset is None and len(self.dataSets) == 1:
            dset = next(iter(self.dataSets.values()))
        if dset is None:
            return None
        matrix = dset.get('_lazy_plot_data', dset.get('_numpy_plot_data'))
        if matrix is None:
            return None
        return (
            matrix,
            dset['_numpy_plot_column_offset'],
            'Bladed memmap' if '_lazy_plot_data' in dset else 'Rust',
        )
    

def isBinary(filename):
    with open(filename, 'r') as f:
        try:
            # first try to read as string
            l = f.readline()
            # then look for weird characters
            for c in l:
                code = ord(c)
                if code<10 or (code>14 and code<31):
                    return True
            return False
        except UnicodeDecodeError:
            return True

if __name__ == '__main__':
    pass
    #filename = r'E:\Work_Google Drive\Bladed_Sims\Bladed_out_binary.$41'
    #Output = BladedFile(filename)
    #df = Output.toDataFrame()
