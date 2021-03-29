from numpy.fft import fft, fftfreq, ifft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal as signal

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

def kmean_label_df(kmeans, df):
    return pd.DataFrame(list(kmeans.labels_), index=df.index, columns = ['label'])

def ifft_equation(freq_loc, freq_value, n):
    deltax= (2*np.pi)/n
    x_values = np.arange(0,2*np.pi,deltax)
    s=np.cos(freq_loc[0]*x_values)*abs(freq_value[0])/n

    for i in range(1,len(freq_loc)):
        phase_shift = np.arctan2(freq_value[i].imag,freq_value[i].real)
        s = s+ 2*((abs(freq_value[i])/n)*np.cos(freq_loc[i]*x_values+phase_shift))

    return s

def kmean_label_df(kmeans, df):
    return pd.DataFrame(list(kmeans.labels_), index=df.index, columns = ['label'])
