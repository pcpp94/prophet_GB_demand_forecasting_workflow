
from typing import Literal

from .external_variables import *
from .default_variables import *

import pandas as pd
import datetime
import numpy as np


def make_forecast_df(granularity: Literal['D', 'M'], category: Literal['Domestic', 'Non-domestic'], default_seasonality=True, initial_date=initial_date, forecast_date=None, **kwargs):

    if forecast_date == None:
        df = read_demand_df(granularity=granularity, category=category)
        final_date = df['ds'].max()
        forecast_date = final_date + forecast_horizon[granularity]
    else:
        forecast_date = forecast_date

    if granularity == 'D':
        if default_seasonality == True:
            df = pd.DataFrame({'ds': pd.date_range(
                start=initial_date, end=forecast_date, freq='D')})
            df['sector'] = category
            temp = make_daily_regressors_df(initial_date=initial_date)[1]
            seasonality = make_daily_seasonality_df(
                initial_date=initial_date)
            df = df.merge(temp, how='left', on='ds').merge(
                seasonality, how='left', on='ds')
        else:
            df = pd.DataFrame({'ds': pd.date_range(
                start=initial_date, end=forecast_date, freq='D')})
            df['sector'] = category
            temp = make_daily_regressors_df(initial_date=initial_date)[1]
            df = df.merge(temp, how='left', on='ds')
    else:
        regressors = {}
        for i in [i for i in ['gdp_dic', 'pop_dictionary', 'cap_regressor'] if i in kwargs.keys()]:
            regressors[i] = kwargs.get(i)
        df = pd.DataFrame({'ds': pd.date_range(
            start=initial_date, end=forecast_date, freq='MS')})
        df['sector'] = category
        monthly_regressors = make_monthly_regressors_df(
            category=category, **regressors)
        df = df.merge(monthly_regressors, how='left', on='ds')

    return df


def make_complete_input_df(granularity: Literal['D', 'M'], category: Literal['Domestic', 'Non-domestic'], default_seasonality=True, **kwargs):

    if granularity == 'D':
        if default_seasonality == True:
            df = read_demand_df(granularity='D', category=category)
            temp = make_daily_regressors_df()[0]
            seasonality = make_daily_seasonality_df()
            df = df.merge(temp, how='inner', on='ds').merge(
                seasonality, how='left', on='ds')
        else:
            df = read_demand_df(granularity='D', category=category)
            temp = make_daily_regressors_df()[0]
            df = df.merge(temp, how='inner', on='ds')
    else:
        regressors = {}
        for i in [i for i in ['gdp_dic', 'pop_dictionary', 'cap_regressor'] if i in kwargs.keys()]:
            regressors[i] = kwargs.get(i)
        df = read_demand_df(granularity='M', category=category)
        monthly_regressors = make_monthly_regressors_df(
            category=category, **regressors)
        df = df.merge(monthly_regressors, how='left', on='ds')

    return df


def read_demand_df(granularity: Literal['D', 'M'], category=Literal['Domestic', 'Non-domestic']):

    if granularity == 'D':
        df = pd.read_parquet(demand_path)
        df = df[(df['sector'] == category)].rename(columns={
            'consumption_gwh': 'y',
            'date': 'ds'
        }).reset_index(drop=True).sort_values('ds')

    else:
        df = pd.read_parquet(corrected_demand_path).rename(
            columns={'settlement_date': 'date'})
        df['date'] = df['date'].apply(lambda dt: dt.replace(day=1))
        df = df.groupby(by=['date', 'sector'])[
            ['total_corrected_consumption_gwh']].sum().reset_index()
        df = df[(df['sector'] == category)].rename(columns={
            'total_corrected_consumption_gwh': 'y',
            'date': 'ds'}).reset_index(drop=True).sort_values('ds')

    return df


__all__ = ["make_forecast_df", "make_complete_input_df", "read_demand_df"]
