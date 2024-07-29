#!/usr/bin/env python

"""
Expected decay
    > Var
    > Confidence interval

Lower bound (optimistic)
CVar (truly pessimistic)
PO (pessimistic)
    > Expected between CVar and Expected decay
"""
import scipy.stats as stats
import numpy as np
import project_utils as pu
import math
from statsmodels.tsa.statespace.varmax import VARMAX


scale, order, trend = 1e4, (1, 1), 'ct'


#### VARIMA time-series
def fit_varima(data, order, trend, scale=None, disp=True):
    """
    Fit VARIMA model to data
    """
    diff_data = data.diff().dropna()
    if scale is None:
        model = VARMAX(diff_data, order=order, trend=trend)
    else:
        model = VARMAX(diff_data*scale, order=order, trend=trend)
    fitted_model = model.fit(disp=disp)
    return fitted_model

def forecast_decay_varima(fitted_model, latest_obs, steps, scale=None):
    """
    Forecast decay rates for areas given fitted_model
    :param fitted_model:
    :param latest_data:
    :param steps:
    :param scale:
    :return:
    """
    forecast = fitted_model.get_forecast(steps=steps)
    if scale is None:
        forecast_values = forecast.predicted_mean + latest_obs
    else:
        forecast_values = (forecast.predicted_mean)/scale + latest_obs
    forecast_values.reset_index(drop=True)
    return forecast_values


#### LSTM Time-series
