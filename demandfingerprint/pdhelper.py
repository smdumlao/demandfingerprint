import pandas as pd
import numpy as np
import sklearn.metrics as metrics

def datetime_transform(df):
    '''
    Transfors a df and then assign the columns as datetime index.
    '''
    dfx = df.T
    dfx.index = pd.to_datetime(dfx.index)
    return dfx

def dfdictExcelReader(excel_file, index_col = None):
    return pd.read_excel(excel_file, sheet_name = None, index_col = index_col)

def dfdictExtractColumn(dfdict, col):
    '''
    Extracts a common column from the dfdict.
    '''
    col_df = []
    for k, v in dfdict.items():
        col_df.append(v[col].to_frame(name=k))
    col_df = pd.concat(col_df, axis=1)
    return col_df

def dfdictExtractRow(dfdict, row):
    '''
    Extracts a common labeled row from the dfdict
    '''
    row_df = []
    for k, v in dfdict.items():
        df = v.T
        row_df.append(df[row].to_frame(name=k))
    row_df = pd.concat(row_df, axis=1)
    return row_df

def year_v_week_pivot(series):
    '''
    Pivots the series into a year x weeks.
    '''
    df = series.to_frame('values')
    df['year'] = df.index.year
    df['week'] = df.index.week
    return pd.pivot_table(df,index='year',columns='week',values='values')


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def r_square(data1, data2):
    residuals = np.array(data1) - np.array(data2)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data1-np.mean(data1))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def r_mean_squared_error(data1, data2):
    return np.sqrt(metrics.mean_squared_error(data1, data2))

def date_range_localized(df, tz_area = 'Asia/Tokyo'):
    '''
    Reconstructs the index of the df with the tz
    '''
    df_index = df.index
    dfx = df.copy()
    dfx.index = pd.date_range(start=df_index[0], end=df_index[-1],
                              freq='H', tz=tz_area)
    return dfx

def resample_index(index, freq):
    """Resamples each day in the daily `index` to the specified `freq`.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The daily-frequency index to resample
    freq : str
        A pandas frequency string which should be higher than daily

    Returns
    -------
    pd.DatetimeIndex
        The resampled index

    From: https://stackoverflow.com/questions/37853623/how-to-efficiently-resample-a-datetimeindex
    """
    assert isinstance(index, pd.DatetimeIndex)
    start_date = index.min()
    end_date = index.max() + pd.DateOffset(days=1)
    resampled_index = pd.date_range(start_date, end_date, freq=freq)[:-1]
    series = pd.Series(resampled_index, resampled_index.floor('D'))
    return pd.DatetimeIndex(series.loc[index].values)
