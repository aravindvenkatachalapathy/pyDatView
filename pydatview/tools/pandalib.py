import pandas as pd
import numpy as np
import re


def pd_interp1(x_new, xLabel, df):
    """ Interpolate a panda dataframe based on a set of new value
    This function assumes that the dataframe is a simple 2d-table
    """
    from .signal_analysis import multiInterp
    x_old = df[xLabel].values
    data_new=multiInterp(x_new, x_old, df.values.T)
    return pd.DataFrame(data=data_new.T, columns=df.columns.values)
    #nRow,nCol = df.shape
    #nRow = len(xnew)
    #data = np.zeros((nRow,nCol))
    #xref =df[xLabel].values.astype(float)
    #for col,i in zip(df.columns.values,range(nCol)):
    #    yref = df[col].values
    #    if yref.dtype!=float:
    #        raise Exception('Wrong type for yref, consider using astype(float)')
    #    data[:,i] = np.interp(xnew, xref, yref)
    #return pd.DataFrame(data=data, columns = df.columns)

def create_dummy_dataframe(size):
    return pd.DataFrame(data={'col1': np.linspace(0,1,size), 'col2': np.random.normal(0,1,size)})



def remap_df(df, ColMap, bColKeepNewOnly=False, inPlace=False, dataDict=None, verbose=False):
    """ 
    NOTE: see welib.fast.postpro

    Add/rename columns of a dataframe, potentially perform operations between columns

    dataDict: dictionary of data to be made available as "variable" in the column mapping
         'key' (new) : value (old)

    Example:

        ColumnMap={
          'WS_[m/s]'         : '{Wind1VelX_[m/s]}'             , # create a new column from existing one
          'RtTSR_[-]'        : '{RtTSR_[-]} * 2  +  {RtAeroCt_[-]}'    , # change value of column
          'RotSpeed_[rad/s]' : '{RotSpeed_[rpm]} * 2*np.pi/60 ', # new column [rpm] -> [rad/s]
          'q_p' :  ['Q_P_[rad]', '{PtfmSurge_[deg]}*np.pi/180']  # List of possible matches
        }
        # Read
        df = weio.read('FASTOutBin.outb').toDataFrame()
        # Change columns based on formulae, potentially adding new columns
        df = fastlib.remap_df(df, ColumnMap, inplace=True)

    """
    # Insert dataDict into namespace
    if dataDict is not None:
        for k,v in dataDict.items():
            exec('{:s} = dataDict["{:s}"]'.format(k,k))


    if not inPlace:
        df=df.copy()
    ColMapMiss=[]
    ColNew=[]
    RenameMap=dict()
    # Loop for expressions
    for k0,v in ColMap.items():
        k=k0.strip()
        if type(v) is not list:
            values = [v]
        else:
            values = v
        Found = False
        for v in values:
            v=v.strip()
            if Found:
                break # We avoid replacing twice
            if v.find('{')>=0:
                # --- This is an advanced substitution using formulae
                search_results = re.finditer(r'\{.*?\}', v)
                expr=v
                if verbose:
                    print('Attempt to insert column {:15s} with expr {}'.format(k,v))
                # For more advanced operations, we use an eval
                bFail=False
                for item in search_results:
                    col=item.group(0)[1:-1]
                    if col not in df.columns:
                        ColMapMiss.append(col)
                        bFail=True
                    expr=expr.replace(item.group(0),'df[\''+col+'\']')
                #print(k0, '=', expr)
                if not bFail:
                    df[k]=eval(expr)
                    ColNew.append(k)
                else:
                    print('[WARN] Column not present in dataframe, cannot evaluate: ',expr)
            else:
                #print(k0,'=',v)
                if v not in df.columns:
                    ColMapMiss.append(v)
                    if verbose:
                        print('[WARN] Column not present in dataframe: ',v)
                else:
                    if k in RenameMap.keys():
                        print('[WARN] Not renaming {} with {} as the key is already present'.format(k,v))
                    else:
                        RenameMap[k]=v
                        Found=True

    # Applying renaming only now so that expressions may be applied in any order
    for k,v in RenameMap.items():
        if verbose:
            print('Renaming column {:15s} > {}'.format(v,k))
        k=k.strip()
        iCol = list(df.columns).index(v)
        df.columns.values[iCol]=k
        ColNew.append(k)
    df.columns = df.columns.values # Hack to ensure columns are updated

    if len(ColMapMiss)>0:
        print('[FAIL] The following columns were not found in the dataframe:',ColMapMiss)
        #print('Available columns are:',df.columns.values)

    if bColKeepNewOnly:
        ColNew = [c for c,_ in ColMap.items() if c in ColNew]# Making sure we respec order from user
        ColKeepSafe = [c for c in ColNew if c in df.columns.values]
        ColKeepMiss = [c for c in ColNew if c not in df.columns.values]
        if len(ColKeepMiss)>0:
            print('[WARN] Signals missing and omitted for ColKeep:\n       '+'\n       '.join(ColKeepMiss))
        df=df[ColKeepSafe]
    return df

_UNIT_SCALINGS = {
    'WE': {
        'rad/s': (30 / np.pi, 'rpm'),
        'rad': (180 / np.pi, 'deg'),
        'n': (1e-3, 'kN'),
        'mn': (1e3, 'kN'),
        'nm': (1e-3, 'kNm'),
        'n-m': (1e-3, 'kNm'),
        'n*m': (1e-3, 'kNm'),
        'mnm': (1e3, 'kNm'),
        'mn-m': (1e3, 'kNm'),
        'mn*m': (1e3, 'kNm'),
        'w': (1e-3, 'kW'),
        'mw': (1e3, 'kW'),
    },
    'SI': {
        'rpm': (np.pi / 30, 'rad/s'),
        'deg': (np.pi / 180, 'rad'),
        'deg/s': (np.pi / 180, 'rad/s'),
        'mn': (1e6, 'N'),
        'kn': (1e3, 'N'),
        'mnm': (1e6, 'Nm'),
        'mn-m': (1e6, 'Nm'),
        'mn*m': (1e6, 'Nm'),
        'knm': (1e3, 'Nm'),
        'kn-m': (1e3, 'Nm'),
        'kn*m': (1e3, 'Nm'),
        'kw': (1e3, 'W'),
        'mw': (1e6, 'W'),
    },
}


def _split_unit(column_name):
    """Return the variable, unit, separator, and bracket style."""
    column_name = str(column_name)
    square = column_name.rfind('[')
    round_ = column_name.rfind('(')
    start = max(square, round_)
    if start <= 1:
        return column_name, '', '', ''
    closing = ']' if start == square else ')'
    end = column_name.find(closing, start + 1)
    if end < 0:
        return column_name, '', '', ''
    separator = column_name[start - 1:start]
    if separator in (' ', '_'):
        variable = column_name[:start - 1]
    else:
        separator = ''
        variable = column_name[:start]
    brackets = '[]' if start == square else '()'
    return variable, column_name[start + 1:end], separator, brackets


def unitConversionPlan(columns, flavor='SI'):
    """Return renamed columns and only the numeric scalings that are needed."""
    if flavor not in _UNIT_SCALINGS:
        raise NotImplementedError(flavor)
    scalings = _UNIT_SCALINGS[flavor]
    renamed = []
    conversions = []
    for index, column_name in enumerate(columns):
        variable, unit, separator, brackets = _split_unit(column_name)
        conversion = scalings.get(unit.lower())
        if conversion is None:
            renamed.append(str(column_name))
            continue
        scale, new_unit = conversion
        renamed.append(
            variable + separator + brackets[0] + new_unit + brackets[1]
        )
        conversions.append((index, scale))
    return renamed, conversions


def changeUnits(df, flavor='SI', inPlace=True, plan=None):
    """Change dataframe units in place, skipping columns that need no scaling."""
    if not inPlace:
        raise NotImplementedError()
    renamed, conversions = plan or unitConversionPlan(df.columns, flavor)
    for index, scale in conversions:
        scaled = df.iloc[:, index] * scale
        if hasattr(df, 'isetitem'):
            df.isetitem(index, scaled)
        else:
            df.iloc[:, index] = scaled
    if list(df.columns) != renamed:
        df.columns = renamed
    return df
