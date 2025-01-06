from supabase_client.utils import get_client
import pandas as pd
import numpy as np
import streamlit as st
from statsmodels.tsa.stattools import adfuller, acf
from datetime import timedelta

@st.cache_data(ttl=timedelta(hours=1))
def fetch_data():
    client = get_client()
    response = (client
                .table('weather_data')
                .select('*')
                .order('time')
                .execute())
    df = pd.DataFrame(data=response.data)
    df.set_index('time',drop=True,inplace=True)
    return df

@st.cache_data(ttl=timedelta(hours=1))
def fetch_forecast():
    client = get_client()
    response = (client
                .table('weather_forecast')
                .select('*')
                .order('time', desc=True)
                .limit(6)
                .execute())
    df = pd.DataFrame(data=response.data)
    df.set_index('time',drop=True,inplace=True)
    return df

@st.cache_data(ttl=timedelta(hours=1))
def adfuller_test(_data:pd.DataFrame):
    data1 = _data.dropna()
    out = []
    for col in data1.columns:
        result = adfuller(data1[col],maxlag=20)
        statistic = result[0]
        pvalue = result[1]
        lags = result[2]
        stationary = 'Stationary' if pvalue < 0.05 else 'Non-Stationary'
        out.append((col,round(statistic,4),round(pvalue,4),lags, stationary))
    return out

@st.cache_data(ttl=timedelta(hours=1))
def calculate_acf(_data:pd.DataFrame):
    results = []
    for col in _data.columns:
        res = acf(_data[col],adjusted=True,nlags=20,alpha=0.05,missing='drop')[0]
        results.append(res)
    return results