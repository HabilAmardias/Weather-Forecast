import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, VECMResults, coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR, VARResultsWrapper
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error
from pandas import Index
import optuna
from prefect import flow, task
from supabase_client.utils import get_client

@task
def load_data():
    client = get_client()
    response = client.table('weather_data').select('*').execute()
    df = pd.DataFrame(data=response.data)
    return df

@task
def clean_data(df:pd.DataFrame):
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time',inplace=True)
    df.set_index('time',drop=True,inplace=True)
    
    deltas = df.index.diff()[1:]
    gaps = deltas[deltas > timedelta(days=1)]

    if len(gaps) > 0:
        df = df.resample('D').interpolate(method='time')

    if df.isna().sum().values.sum().item() > 0:
        df = df.interpolate(method='time')

    return df

@task
def split_data(df:pd.DataFrame, n_test:int):
    train, test = df.iloc[:-n_test], df.iloc[-n_test:]
    return train, test

@task
def cointegration_test(df:pd.DataFrame, det_order:int):
    if det_order not in [-1,0,1]:
        raise ValueError('Wrong Deterministic Order')
    
    var = VAR(df)
    lag_order_result = var.select_order(20)

    chosen_lag = lag_order_result.selected_orders['aic'].item() - 1
    test_result = coint_johansen(endog = df, det_order=det_order, 
                   k_ar_diff=chosen_lag)
    
    traces = test_result.lr1
    critical_value = test_result.cvt[:,1] #95% crit values

    r = 0
    for i in range(len(traces)):
        if traces[i] <= critical_value[i]:
            break
        r += 1
    
    return r, chosen_lag

@task
def fit_forecast_vecm(df:pd.DataFrame, det_order:int, n_test:int, test_index:Index):

    r, chosen_lag = cointegration_test(df, det_order)
    vecm: VECMResults = VECM(df,coint_rank=r,k_ar_diff=chosen_lag).fit()

    preds = vecm.predict(steps = n_test)
    preds_df = pd.DataFrame(preds, columns=df.columns, index=test_index)

    return vecm, preds_df

@task
def stationary_test(df:pd.DataFrame):
    stationary_columns=[]
    non_stationary_columns=[]

    for col in df.columns:
        test_result = adfuller(df[col])
        pvalue = test_result[1]

        if pvalue > 0.05:
            non_stationary_columns.append(col)
        else:
            stationary_columns.append(col)
    
    return stationary_columns, non_stationary_columns

@task
def differencing_data(df:pd.DataFrame, non_stationary_columns:list[str]):

    if len(non_stationary_columns) == 0:
        return df
    
    differenced = df.copy(deep=True)
    
    for col in df.columns:
        if col in non_stationary_columns:
            differenced[col] = differenced[col].diff()

    differenced.dropna(inplace=True)
    return differenced

@task
def inverse_difference(pred:pd.DataFrame, df:pd.DataFrame, non_stationary_columns:list[str]):

    if len(non_stationary_columns) == 0:
        return pred
    
    for col in pred.columns:
        if col in non_stationary_columns:
            pred[col] = df[col].iloc[-1] + pred[col].cumsum()
    
    return pred

@task
def fit_forecast_var(df:pd.DataFrame, n_test:int, test_index:Index):

    stationary_columns, non_stationary_columns = stationary_test(df)
    differenced = differencing_data(df, non_stationary_columns)

    var = VAR(differenced)
    lag_order_result = var.select_order(20)
    chosen_lag = lag_order_result.selected_orders['aic'].item()

    var_result:VARResultsWrapper = var.fit(chosen_lag)
    lags = var_result.k_ar

    forecast_input = differenced.iloc[-lags:].values
    preds:np.ndarray = var_result.forecast(forecast_input,steps = n_test)
    preds_df = pd.DataFrame(preds, columns=df.columns, index = test_index)

    return var_result, inverse_difference(preds_df, df, non_stationary_columns)

@task
def build_model(model_name:str, train:pd.DataFrame, test:pd.DataFrame | None, n_test:int, det_order:int):

    if isinstance(test, pd.DataFrame):
        index = test.index
    else:
        start = train.index[-1] + timedelta(days=1)
        end = start + timedelta(days=n_test-1)
        index = pd.date_range(
            start = start,
            end = end,
            freq='D'
        )
    
    if model_name == 'vecm':
        model, preds = fit_forecast_vecm(train, det_order, n_test, index)
    elif model_name == 'var':
        model, preds = fit_forecast_var(train, n_test, index)
    else:
        raise ValueError('Wrong Model Name')

    return model, preds

@task
def hyperparameter_tuning(df:pd.DataFrame):
    
    def objective(trial:optuna.trial.Trial):
        model_name = trial.suggest_categorical('model_name', ['vecm','var'])
        det_order = trial.suggest_int('det_order',low=-1, high=1, step=1)

        train, test = split_data(df, n_test=6)

        model,preds = build_model(model_name=model_name,
                                  train=train,
                                  test=test,
                                  n_test=6,
                                  det_order=det_order)
        
        error = 0
        for col in preds.columns:
            error += mean_absolute_percentage_error(
                test[col],
                preds[col]
            )
        error /= len(preds.columns)

        return error
    
    search_space = {'model_name': ['vecm','var'], 'det_order': [-1, 0, 1]}
    sampler=optuna.samplers.GridSampler(search_space=search_space,seed=42)
    study=optuna.create_study(sampler=sampler,direction='minimize')
    study.optimize(objective)

    model_name = study.best_params.get('model_name','var')
    det_order = study.best_params.get('det_order', 0)

    return model_name, det_order, study.best_value

@task
def log_forecast(df:pd.DataFrame):
    records = df.to_dict('records')
    client = get_client()
    client.table('weather_forecast').upsert(records).execute()

@flow
def forecast_pipeline():
    df = load_data()
    df = clean_data(df)

    model_name, det_order, error = hyperparameter_tuning(df)

    model, forecast = build_model(model_name=model_name,
                                  train=df,
                                  test=None,
                                  n_test=6,
                                  det_order=det_order)
    
    forecast.reset_index(drop=False, inplace=True, names=['time'])
    forecast['time'] = forecast['time'].dt.strftime('%Y-%m-%d')

    log_forecast(forecast)

if __name__ == '__main__':
    forecast_pipeline.serve(
        name='forecast-pipeline',
        interval=timedelta(days=5)
    )
    

    
    
    
    
