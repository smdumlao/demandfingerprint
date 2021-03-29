import pandas as pd

offset_color = ['crimson',"darkorange",'k', "royalblue" , "darkviolet"]
offset_cols = ['warmpeak', 'warmmean', 'magmean', 'coldmean', 'coldpeak']

def calc_temp_offset(temp, amax=22, amin=18):
    '''
    Returns the distance from the acceptable max ``amax`` and
    acceptable min ``amin``.
    '''
    if temp >= amax:
        return temp - amax
    elif amin < temp < amax:
        return 0
    else:
        return temp - amin

def get_temp_offset_stat(temp, year, reset_cols=True, maincol=True):
    '''
    Calcui
    '''

    temp_offset = temp[str(year)].apply(calc_temp_offset).to_frame('offset')
    temp_offset['mag'] = temp_offset['offset'].where(temp_offset['offset'] > 0, -temp_offset['offset'])
    temp_offset['warm'] = temp_offset['offset'].where(temp_offset['offset'] > 0, 0)
    temp_offset['cold'] = temp_offset['offset'].where(temp_offset['offset'] < 0, 0)

    temp_offset_stat = temp_offset.mag.resample('M').mean().to_frame('magmean')
    temp_offset_stat['warmmean'] = temp_offset.warm.resample('M').mean()
    temp_offset_stat['warmstd'] = temp_offset.warm.resample('M').std()
    temp_offset_stat['warmpeak'] = temp_offset.warm.resample('M').max()
    temp_offset_stat['coldmean'] = temp_offset.cold.resample('M').mean()
    temp_offset_stat['coldstd'] = temp_offset.warm.resample('M').std()
    temp_offset_stat['coldpeak'] = temp_offset.cold.resample('M').min()

    if reset_cols:
        temp_offset_stat.index = list(range(1,13))
    if maincol:
        temp_offset_stat = temp_offset_stat[offset_cols]
    return temp_offset_stat

def eval_year(temp, col_month=False):
    '''
    Evaluates a year
    '''
    data = []
    year_max = temp.resample('D').max()
    year_min = temp.resample('D').min()

    data.append(temp.resample('M').mean())

    data.append(year_max.resample('M').max())
    data.append(year_max.resample('M').mean())
    data.append(year_max.resample('M').min())

    data.append(year_min.resample('M').max())
    data.append(year_min.resample('M').mean())
    data.append(year_min.resample('M').min())

    index = ['mean', 'maxmax', 'maxmean', 'maxmin', 'minmax', 'minmean', 'minmin']

    df = pd.DataFrame(data, index=index)

    if col_month:
        df.columns = range(1,13)
    return df
