import pandas as pd
import numpy as np
from datetime import date, timedelta

def getnextmonday(date0):
    d = date0
    while(not d.weekday()==0):
        d += timedelta(days = 1)
    return d

def getprevmonday(date0):
    d = date0
    while(not d.weekday()==0):
        d += timedelta(days = -1)
    return d

def getprevsunday(date0):
    d = date0
    while(not d.weekday()==6):
        d += timedelta(days = -1)
    return d
