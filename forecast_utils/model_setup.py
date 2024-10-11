from typing import Literal

import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from hyperopt import fmin, tpe, hp, anneal, Trials, SparkTrials, STATUS_OK, space_eval
from .default_variables import *


def create_hyperopt_space(model, n_iter=None, algo=tpe.suggest, random_state=42, **kwargs):

    if len(model.extra_regressors) == 0:
        raise Exception(
            "Regressors and Seasonalities (if needed) need to be added to the forecaster")

    if model.granularity == 'D':
        base_keys = daily_base_tuning
        default_base_params = daily_base_hyperopt
        regressors_keys = [i for i in model.extra_regressors]
        regressors_params = daily_regressors_hyperopt
        if n_iter is None:
            n_iter = 50
        else:
            n_iter = n_iter
    else:
        base_keys = monthly_base_tuning
        default_base_params = monthly_base_hyperopt
        regressors_keys = [i for i in model.extra_regressors]
        if n_iter is None:
            n_iter = 100
        else:
            n_iter = n_iter
        if model.category == 'Domestic':
            regressors_params = monthly_dom_regressors
        else:
            regressors_params = monthly_non_dom_regressors

    space = {}
    space_dict = {}

    for i in base_keys:
        if kwargs.get(i) is None:
            space[i] = hp.uniform(i, default_base_params[i]
                                  [0], default_base_params[i][1])
            space_dict[i] = ('hp.uniform', default_base_params[i]
                             [0], default_base_params[i][1])
        else:
            space[i] = hp.uniform(i, kwargs.get(i)[0], kwargs.get(i)[1])
            space_dict[i] = ('hp.uniform', kwargs.get(i)[0], kwargs.get(i)[1])

    for i in regressors_keys:
        if kwargs.get(i) is None:
            space[i] = hp.uniform(i, regressors_params[i]
                                  [0], regressors_params[i][1])
            space_dict[i] = ('hp.uniform', regressors_params[i]
                             [0], regressors_params[i][1])
        else:
            space[i] = hp.uniform(i, kwargs.get(i)[0], kwargs.get(i)[1])
            space_dict[i] = ('hp.uniform', kwargs.get(i)[0], kwargs.get(i)[1])

    space = space
    space_dict = space_dict

    return (space, space_dict)


def scoring_outputs(model):

    if model.history is None:
        raise Exception('Model has not been fit.')

    if model.granularity == 'D':
        df_cv = cross_validation(model, initial=daily_initial,
                                 period=daily_period, horizon=daily_cross_val_horizon)
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['cutoff_points'] = len(df_cv['cutoff'].unique())
        score = df_p['mape'].values[0]
        return (df_cv, df_p, score)
    else:
        df_cv = cross_validation(model, initial=monthly_initial,
                                 period=monthly_period, horizon=monthly_cross_val_horizon)
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['cutoff_points'] = len(df_cv['cutoff'].unique())
        score = df_p['mape'].values[0]
        return (df_cv, df_p, score)


def default_prophet_model(granularity, category, holidays=None):

    if granularity == "D":
        model = Prophet(  # Default values
            growth='linear',
            seasonality_mode='multiplicative',
            holidays_mode='multiplicative',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            holidays=holidays,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
    else:
        model = Prophet(  # Default values
            growth='linear',
            seasonality_mode='multiplicative',
            holidays_mode='multiplicative',
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.9,
            holidays=None,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )

    model.granularity = granularity
    model.category = category

    return model


__all__ = ["create_hyperopt_space", "scoring_outputs", "default_prophet_model"]
