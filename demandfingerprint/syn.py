"""
The ``syn`` module contains functions for synthesizing demand based on
available hourly data by creating a fingerprint model of demand
and generating the demand based on the temperature data.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

import scipy
import scipy.signal as signal
from numpy.fft import fft

import pdhelper
import datehelper

def get_peaktrough(series_raw, prominence=0.2):
    '''
    Identifies the peaks and troughs in the given series and returns it as
    a dataframe.
    '''
    if isinstance(series_raw, list):
        series = series_raw
    else:
        series = series_raw.to_list()

    peak_indicies = signal.find_peaks(series, prominence=prominence)

    peak_loc = []
    for peak in peak_indicies[0]:
        peak_loc.append(series[peak])

    trough = [-x for x in series]
    trough_indicies = signal.find_peaks(trough, prominence=prominence)

    trough_loc = [series[0]]
    for trough in trough_indicies[0]:
        trough_loc.append(series[trough])

    p_df = pd.DataFrame(peak_loc, index=peak_indicies[0], columns = ['val'])
    p_df['type'] = 0 #0 for peaks

    t_df = pd.DataFrame(trough_loc, index=[0] + list(trough_indicies[0])
        , columns = ['val'])
    t_df['type'] = 1 #1 for troughs

    merge_df = pd.concat([p_df, t_df])
    merge_df = merge_df.sort_index()
    return merge_df

def get_fingerprint(series, peaktrough_df, dfreturn = True):
    '''
    Get the fingerprint of the demand.

    The peaks or troughs are identified and selected as representative for
    0, 3, 6, 15, 18, 21, 24. Actual values are selected for 9, 11, 12, 13
    '''

    #peaks per day
    fingerprintA = list(range(0,25,3))
    fingerprintA.remove(12)
    fingerprintA.remove(9)
    fingerprintB = [9,11,12,13]

    #peaks per week
    fingerprintA_week = []
    for i in range(7):
        fingerprintA_week.extend([(24*i) + f for f in fingerprintA[:-1]])

    fingerprintB_week = []
    for i in range(7):
        fingerprintB_week.extend([(24*i) + f for f in fingerprintB])

    fingerprintB_week.append(167)

    #get the samples based on fingerprint A
    fingerprint_sample = []
    for time0 in fingerprintA_week:
        range_t = list(range(time0, time0+3))
        dfx = peaktrough_df[peaktrough_df.index.isin(range_t)]
        dfx = dfx.sort_values('type')

        if len(dfx) > 0:
            fingerprint_sample.append(dfx.iloc[0].val)
        else:
            fingerprint_sample.append(series[time0])

    fpA_df = pd.DataFrame(fingerprint_sample, index=fingerprintA_week, columns = ['val'])

    #get the samples based on fingerprint B
    fingerprint_sample = [series[t] for t in fingerprintB_week]
    fpB_df = pd.DataFrame(fingerprint_sample, index=fingerprintB_week, columns = ['val'])

    #merge
    fingerprint_df = pd.concat([fpA_df, fpB_df])
    fingerprint_df = fingerprint_df.sort_index()

    if dfreturn:
        return fingerprint_df
    else:
        return fingerprint_df['val'].to_list()

def reconstruct_from_fingerprint(fingerprint_df, roundval = 2):
    '''
    Reconstruct the data from the weekly fingerprint
    '''
    edges = fingerprint_df.index.to_list()
    edges_val = fingerprint_df.val.to_list()

    new_vals = []
    for i in range(len(edges)-1):
        new_vals.append(edges_val[i])
        for j in range(edges[i]+1, edges[i+1]):
            x = [edges[i], edges[i+1]]
            y = [edges_val[i], edges_val[i+1]]
            new_y = np.interp(j, x, y)
            new_vals.append(round(new_y,roundval))
    new_vals.append(edges_val[-1])

    return new_vals

def align_to_fingerprint_wk(series, roundval = 2):
    '''
    Workflow of the alignment per week.
    '''
    peaktrough_df = get_peaktrough(series)
    fingerprint_df = get_fingerprint(series, peaktrough_df)
    new_vals = reconstruct_from_fingerprint(fingerprint_df, roundval = roundval)
    new_vals = pd.Series(new_vals, index=series.index)
    return new_vals

def align_to_fingerprint(weekly_df):
    '''
    Aligns the data to the defined fingerprint.
    This is important for fft since it requires continuous signals.
    '''
    weekly_df_peaks = weekly_df.T.copy()
    for col in weekly_df_peaks.columns:
        weekly_df_peaks[col] = align_to_fingerprint_wk(weekly_df_peaks[col])
    weekly_df_peaks = pdhelper.datetime_transform(weekly_df_peaks)
    weekly_df_peaks = weekly_df_peaks.applymap(lambda x: round(x, 2))

    return weekly_df_peaks

#-----------------------------------------------------------------------------#
# fft

def z_score_standardization(series):
    '''
    z-transform for pandas dataframe
    '''
    mean = series.mean()
    std = series.std()
    return series.apply(lambda x: (x-mean)/std)

def fft_magnitude(data):
    '''
    Calculates the magnitude of the fft
    '''
    n = len(data)
    fft_vals = fft(data)
    fft_theo = 2.0*np.abs(fft_vals/n)

    return fft_theo

def reconstruct_fingerprint(x, a, b, freq_mag, freq_phase, n=168):
    deltax= (2*np.pi)/168

    # average
    y = ((freq_mag[0])+a)*np.cos(0*x*deltax)

    #daily harmonics
    for i in range(1,8):
        y = y + (2/n)*b*(freq_mag[i])*np.cos(i*7*x*deltax+freq_phase[i])

    return y

def extract_fingerprint(cluster_df):
    '''
    Extract the fingerprint from each cluster by merging all the samples into
    a long list of data. The focus is mainly on the weekday since the general
    pattern of the week can be seen in the weekdays.
    '''
    ## daily harmonics
    # extract the weekday data for the daily harmonics
    weekday = cluster_df[list(range(0,24*5))]
    weekday_data = weekday.values.reshape(1,weekday.size).tolist()[0]

    #location of the daily harmonics
    n = len(weekday_data)
    fn = int(n/24)

    weekly_fft = fft(weekday_data)

    freq_loc = [fn*i for i in range(0,8)]
    freq_value = [weekly_fft[i] for i in freq_loc]
    freq_mag = [(abs(value)/n) for value in freq_value]
    freq_phase = [np.arctan2(value.imag,value.real) for value in freq_value]

    ## Weekly Harmonics
    whole_week_data = cluster_df.values.reshape(1,cluster_df.size).tolist()[0]
    n = len(whole_week_data)
    fn = int(n/(24*7))

    whole_week_fft = fft(whole_week_data)

    freq_loc = [fn*i for i in range(1,7)] #only 6 harmonics
    freq_value = [whole_week_fft[i] for i in freq_loc]
    freq_mag.extend([(abs(value)/n) for value in freq_value])
    freq_phase.extend([np.arctan2(value.imag,value.real) for value in freq_value])

    return freq_mag, freq_phase
#-----------------------------------------------------------------------------#
# Kmeans

def kmean_label_df(kmeans, df):
    return pd.DataFrame(list(kmeans.labels_), index=df.index, columns = ['label'])

#-----------------------------------------------------------------------------#
#

def calculate_weekly_demand_statistics(weekly_df):
    '''
    Calculates the average weekday min max and the sat and sun min max
    '''
    weekday_hours = list(range(0, 120))
    sat_hours = list(range(120, 144))
    sun_hours = list(range(144, 168))

    weekly_power_minmax = []
    weekly_power_minmax.append(weekly_df[weekday_hours].min(axis=1).to_list())
    weekly_power_minmax.append(weekly_df[weekday_hours].max(axis=1).to_list())
    weekly_power_minmax.append(weekly_df[sat_hours].min(axis=1).to_list())
    weekly_power_minmax.append(weekly_df[sat_hours].max(axis=1).to_list())
    weekly_power_minmax.append(weekly_df[sun_hours].min(axis=1).to_list())
    weekly_power_minmax.append(weekly_df[sun_hours].max(axis=1).to_list())

    weekly_power_minmax = pd.DataFrame(weekly_power_minmax, index = ['wkd_min',
    'wkd_max', 'sat_min', 'sat_max', 'sun_min', 'sun_max']).T

    weekly_power_minmax.index = weekly_df.index

    return weekly_power_minmax

def calculate_weekend_ratio(weekly_demand_stat, plot=False):
    '''
    Calculates the weekday to weekend ratio
    '''
    def yearly_pivot(series):
        df = series.to_frame('values')
        df['year'] = df.index.year
        df['week'] = df.index.week
        return pd.pivot_table(df,index='year',columns='week',values='values')

    sat_max_ratio = yearly_pivot(weekly_demand_stat.sat_max/weekly_demand_stat.wkd_max)
    sun_max_ratio = yearly_pivot(weekly_demand_stat.sun_max/weekly_demand_stat.wkd_max)
    sat_min_ratio = yearly_pivot(weekly_demand_stat.sat_min/weekly_demand_stat.wkd_min)
    sun_min_ratio = yearly_pivot(weekly_demand_stat.sun_min/weekly_demand_stat.wkd_min)

    if plot:
        sat_max_ratio.T.mean(axis=1).plot(label='sat_max vs wkd_max');
        sun_max_ratio.T.mean(axis=1).plot(label='sun_max vs wkd_max');
        sat_min_ratio.T.mean(axis=1).plot(label='sat_min vs wkd_max');
        sun_min_ratio.T.mean(axis=1).plot(label='sun_min vs wkd_max');

    week_end_ratio = [sat_min_ratio.mean().mean(), sat_max_ratio.mean().mean(),
                      sun_min_ratio.mean().mean(), sun_max_ratio.mean().mean()]

    return week_end_ratio

#-----------------------------------------------------------------------------#
# nonholiday weekday statistics
def get_jp_holiday(data_dir, year):
    holiday = pd.read_excel(f'{data_dir}/jp_holidays.xlsx', sheet_name=str(year), index_col='Date')
    holiday.index = pd.to_datetime(holiday.index)
    holiday['Date'] = holiday.apply(lambda s: f'{year}-{s.name.month}-{s.name.day}', axis=1)
    holiday.index = pd.to_datetime(holiday['Date'])
    return holiday[['Name', 'Type']]

def get_jp_holidays(data_dir, start_year, end_year, nowork=True, tolist=True):
    holidays = []
    for year in range(start_year, end_year+1):
        holidays.append(get_jp_holiday(data_dir, year))

    holidays = pd.concat(holidays)
    if nowork:
        holidays = holidays[(holidays['Type']=='National holiday') | (holidays['Type']=='Bank holiday')]
    if tolist:
        holidays = holidays.index.strftime('%Y-%m-%d').to_list()
    return holidays

def get_kyushu_nonholiweekday_stat(kyushu_data, kyushu_temp, holidays,
                                   start_date = '2016-04', end_date = '2020-03'):
    '''
    Extract the non-holiday weekday temperature and demand.
    '''

    kyushu_demand = kyushu_data['Demand'][start_date:end_date].to_frame('Demand')
    kyushu_demand['wkd'] = kyushu_demand.index.weekday
    kyushu_demand['temp'] = kyushu_temp['kyushu'][start_date:end_date].copy()

    kyushu_demand_wkd = kyushu_demand[kyushu_demand.wkd < 5]

    kyushu_demand_wkd_daily = kyushu_demand_wkd.Demand.resample('D').max().to_frame('maxdemand')
    kyushu_demand_wkd_daily['mindemand'] = kyushu_demand_wkd.Demand.resample('D').min()
    kyushu_demand_wkd_daily['mintemp'] = kyushu_demand_wkd.temp.resample('D').min()
    kyushu_demand_wkd_daily['maxtemp'] = kyushu_demand_wkd.temp.resample('D').max()

    kyushu_demand_wkd_daily = kyushu_demand_wkd_daily.dropna()

    kyushu_demand_wkd_daily = kyushu_demand_wkd_daily[~kyushu_demand_wkd_daily.index.isin(holidays)]

    return kyushu_demand_wkd_daily

#-----------------------------------------------------------------------------#
# fitting functions
def piecewise_linear3(x, x0, x1, y0, y1, k1, k2, k3):
    condlist = [x <= x0, (x > x0) & (x < x1), x >= x1]
    funclist = [lambda x:k1*x + y0 - k1*x0,
                lambda x: (x-x0)*(y1-y0)/(x1-x0) + y0,
                lambda x:k3*x + y1 - k3*x1]
    return np.piecewise(x, condlist, funclist)

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def quadratic_roots(a, b, c):
    d = np.sqrt(b**2 - 4*a*c)
    x1 = (-b + d)/2*a
    x2 = (-b - d)/2*a
    return (x1, x2)

#-----------------------------------------------------------------------------#
def _chunker(lst, n):
    '''
    Yield successive n-sized chunks from lst.
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _mean_in_chunks(series, func, sizeofchunks=24):
    '''
    Calculates the mean value in chunks of the data.
    '''
    data = series.to_list()
    chunks = list(_chunker(data, sizeofchunks))
    samples = []
    for chunk in chunks:
        samples.append(func(chunk))
    return np.mean(samples)

def get_week_minmax(data):
    '''
    Get the minimum and maximum data for a list of weekly data.
    '''

    chunks = list(_chunker(data, 24))

    minval = 0
    maxval = 0
    for i in range(5):
        minval += min(chunks[i])
        maxval += max(chunks[i])
    minval = minval/5
    maxval = maxval/5

    sat_min = min(chunks[5])
    sat_max = max(chunks[5])
    sun_min = min(chunks[6])
    sun_max = max(chunks[6])

    return minval, maxval, sat_min, sat_max, sun_min, sun_max

def get_kyushu_weekly_minmax(hourly_df, start_date=None,
                            end_date = None, target_col='kyushu'):
    '''
    Returns the mean minimum and maximum temperature for the provided
    DataFrame or Series.
    '''

    temp_weekly = get_weekly(hourly_df)

    weekly_df = temp_weekly.apply(_mean_in_chunks,
                               args=(np.min,), axis=1).to_frame('mintemp')
    weekly_df['maxtemp'] = temp_weekly.apply(_mean_in_chunks,
                               args=(np.max,), axis=1)
    return weekly_df

def get_weekday_hourly_apply(weekly_df, applyfx, hour_no):
    '''
    apply a function to the weekly df
    '''
    hour_samples = [(hour_no)+(i*24) for i in range (5)]
    return weekly_df[hour_samples].apply(lambda s: applyfx(s), axis=1)


#-----------------------------------------------------------------------------#
# generate demand
def generate_temp_ref(year, kyushu_temp_db):
    '''
    Generate a index based on the specified year.
    '''
    first_monday = datehelper.getprevmonday(date(year,1,1))
    year_end = date(year,12,31)
    last_monday = year_end + timedelta(days=-(year_end.weekday()))
    sim_end = last_monday + timedelta(days=6)

    return kyushu_temp_db[first_monday:sim_end.strftime('%Y-%m-%d')]

def get_weekly(series_raw, addweeks = 1):
    '''
    Restructure the data to fit into weekly
    '''

    first_monday = datehelper.getnextmonday(series_raw.index[0].date())
    last_sunday = datehelper.getprevsunday(series_raw.index[-1].date())
    series = series_raw[first_monday:last_sunday+timedelta(days=addweeks*7)]

    # group to weeks
    rows0 = len(series)
    cols = 168
    rows = int(rows0/cols)

    narray = series.values[:cols*rows]
    narray = narray.reshape(rows, cols)
    series_weekly = pd.DataFrame(narray)

    new_index = series.resample('W').sum().index.shift(-6, freq='D')[:rows]
    series_weekly.index = new_index

    return series_weekly

def get_weekly_stats(syn_temp, syn_temp_weekly):
    weekday_col = list(range(0, 24*7))

    first_monday = syn_temp.index[0]
    sim_end = syn_temp.index[-1]+timedelta(days=6)

    syn_temp_weekly_stat = get_kyushu_weekly_minmax(syn_temp,
                                                    start_date = first_monday,
                                                    end_date = sim_end)

    hour_samples = [6+(i*24) for i in range(5)]
    syn_temp_weekly_stat['6AM_min'] = syn_temp_weekly[hour_samples].min(axis=1)

    hour_samples = [9+(i*24) for i in range(5)]
    syn_temp_weekly_stat['9AM_max'] = syn_temp_weekly[hour_samples].max(axis=1)

    hour_samples = [15+(i*24) for i in range(5)]
    syn_temp_weekly_stat['3PM_min'] = syn_temp_weekly[hour_samples].min(axis=1)

    for i in range(5):
        hour_samples = [j+(i*24) for j in range(24)]
        syn_temp_weekly_stat[f'd{i}min'] = syn_temp_weekly[hour_samples].min(axis=1)
        syn_temp_weekly_stat[f'd{i}max'] = syn_temp_weekly[hour_samples].max(axis=1)

    syn_temp_weekly_stat['month'] = syn_temp_weekly_stat.index.month

    return syn_temp_weekly_stat

def generate_daily_minmax_demand(syn_temp_weekly_stat, week_end_ratio,
        mindemand_func, maxdemand_func):
    '''
    Prepare the data needed for the demand construction
    '''

    daily_minmax_demand = mindemand_func(syn_temp_weekly_stat.mintemp).to_frame('mindemand')
    daily_minmax_demand['maxdemand'] = maxdemand_func(syn_temp_weekly_stat.maxtemp.values)

    daily_minmax_demand['sat_min'] = daily_minmax_demand['mindemand']*week_end_ratio[0]
    daily_minmax_demand['sat_max'] = daily_minmax_demand['maxdemand']*week_end_ratio[1]
    daily_minmax_demand['sun_min'] = daily_minmax_demand['mindemand']*week_end_ratio[2]
    daily_minmax_demand['sun_max'] = daily_minmax_demand['maxdemand']*week_end_ratio[3]

    for i in range(5):
        daily_minmax_demand[f'd{i}min'] = mindemand_func(syn_temp_weekly_stat[f'd{i}min'])
        daily_minmax_demand[f'd{i}max'] = maxdemand_func(syn_temp_weekly_stat[f'd{i}max'].values)

    return daily_minmax_demand

def assign_weekly_fingerprint(syn_temp_weekly_stat, knn):
    # predict using kNN
    data_col = ['mintemp', 'maxtemp', '6AM_min','9AM_max',  '3PM_min', 'month']

    y_pred = knn.predict(syn_temp_weekly_stat[data_col])
    y_pred = pd.DataFrame(y_pred, index = syn_temp_weekly_stat.index,
                            columns = ['shape'])
    return y_pred

def fingerprint_fit(values_xknown, values_yknown, freq_mag, freq_phase):
    '''
    Get the coefficient a and b for the daily harmonics.
    '''
    values_yknown = np.array(values_yknown)

    #fitting function
    def fingerprint_fit1(x, a, b):
        return reconstruct_fingerprint(x, a, b, freq_mag, freq_phase)

    weekday_x = [x0 for x0 in values_xknown if x0 < 120]
    weekday_y = values_yknown[:len(weekday_x)]
    coefficients = scipy.optimize.curve_fit(fingerprint_fit1, weekday_x,
                                            weekday_y)
    r1_a, r1_b = coefficients[0]

    return r1_a, r1_b

def reconstruct_week(fp_id, values_yknown, cluster_fingerprint):
    '''

    '''
    values_xknown = list(range(168))
    #get fingerprint
    freq_mag, freq_phase, minloc, maxloc = cluster_fingerprint[fp_id]

    #get the params for fitting
    values_xknown0 = []
    for d in range(7):
        values_xknown0.append(minloc+(d*24))
        values_xknown0.append(maxloc+(d*24))

    fit_y = []
    for values_yknown0 in pdhelper.chunks(list(values_yknown), 2):
        yknown = np.array(values_yknown0*7)
        a, b = fingerprint_fit(values_xknown0, yknown, freq_mag, freq_phase)
        fit_day = reconstruct_fingerprint(np.array(values_xknown), a, b, freq_mag, freq_phase)
        fit_y.extend(fit_day[:24])
    return fit_y

def construct_demand(daily_minmax_demand, weekly_fingerprint, cluster_fingerprint):
    '''

    '''
    syn_demand_fitting_param = pd.concat([daily_minmax_demand,
                                                weekly_fingerprint], axis=1)
    start_index = syn_demand_fitting_param.index[0].strftime('%Y-%m-%d')
    periods = len(syn_demand_fitting_param)*7*24
    datetimeindex = pd.date_range(start_index, periods = periods, freq='H')

    syn_demand = []
    yval_col = []
    for i in range(5):
        yval_col.append(f'd{i}min')
        yval_col.append(f'd{i}max')
    yval_col.extend(['sat_min', 'sat_max', 'sun_min', 'sun_max'])

    for _, wksample in syn_demand_fitting_param.iterrows():
        fp_id = int(wksample['shape'])
        values_yknown0 = wksample[yval_col]
        syn_demand.extend(reconstruct_week(fp_id, values_yknown0, cluster_fingerprint))
    syn_demand = pd.DataFrame(syn_demand, index = datetimeindex, columns = ['Demand'])
    syn_demand.index = pd.to_datetime(syn_demand.index)

    return syn_demand

def generate_demand(syn_temp, knn, mindemand_func,
                    maxdemand_func, week_end_ratio, cluster_fingerprint):
    '''
    Generate demand based on the given temperature.
    ``syn_temp`` reference dataframe
    ``knn`` model for the shape
    '''

    #weekly stat generator
    syn_temp_weekly = get_weekly(syn_temp)
    syn_temp_weekly_stat = get_weekly_stats(syn_temp, syn_temp_weekly)

    #demand ftting param generator
    daily_minmax_demand = generate_daily_minmax_demand(syn_temp_weekly_stat,
        week_end_ratio, mindemand_func, maxdemand_func)
    weekly_fingerprint = assign_weekly_fingerprint(syn_temp_weekly_stat,
        knn)

    syn_demand = construct_demand(daily_minmax_demand, weekly_fingerprint,
        cluster_fingerprint)

    return syn_demand

#-----------------------------------------------------------------------------#
# additional code for testing

def get_minmax_test(actual_demand, syn_temp):
    '''
    Used for testing the minmax correlation function.

    Extracts the minmax from the actual data from the EPCO.
    '''

    demand = actual_demand[syn_temp.index[0]:syn_temp.index[-1]]
    demand_weekly = get_weekly(demand)

    i=5
    hour_samples = [j+(i*24) for j in range(24)]
    daily_minmax_demand = demand_weekly[hour_samples].min(axis=1).to_frame('sat_min')
    daily_minmax_demand['sat_max'] = demand_weekly[hour_samples].max(axis=1)

    i=6
    hour_samples = [j+(i*24) for j in range(24)]
    daily_minmax_demand['sun_min'] = demand_weekly[hour_samples].min(axis=1)
    daily_minmax_demand['sun_max'] = demand_weekly[hour_samples].max(axis=1)

    for i in range(5):
        hour_samples = [j+(i*24) for j in range(24)]
        daily_minmax_demand[f'd{i}min'] = demand_weekly[hour_samples].min(axis=1)
        daily_minmax_demand[f'd{i}max'] = demand_weekly[hour_samples].max(axis=1)

    return daily_minmax_demand

def extract_weekly_data(hourly_df, target_col = 'Demand'):
    '''
    Returns the weekly information about the hourly data.
    '''
    first_monday = datehelper.getnextmonday(hourly_df.index[0].date())
    last_sunday = datehelper.getprevsunday(hourly_df.index[-1].date())
    last_monday = datehelper.getprevmonday(last_sunday)
    week_count = int((last_monday - first_monday).days/7)+1

    weekly_data = []
    weekly_mag = []
    for i in range(week_count):
        data = []
        #data_minmax = []
        mon = first_monday + timedelta(days=7*i)
        mon = mon.strftime('%Y-%m-%d')
        sun = first_monday + timedelta(days=(7*i) +6)
        sun = sun.strftime('%Y-%m-%d')
        data.append(mon)
        #data_minmax.append(mon)

        week_df = hourly_df[mon:sun][target_col]
        data.extend(week_df.to_list())
        weekly_data.append(data)

        weekly_df = pd.DataFrame(weekly_data)
        weekly_df = weekly_df.set_index(0)
        weekly_df.index = pd.to_datetime(weekly_df.index)
        weekly_df.index.name='date'
        weekly_df = weekly_df.applymap(lambda x: str(x).replace(',', ''))
        weekly_df = weekly_df.astype('float32')

    return weekly_df
