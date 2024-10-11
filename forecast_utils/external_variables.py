from typing import Literal
import pandas as pd
import numpy as np
import datetime
import calendar
from .default_variables import *


def make_daily_regressors_df(initial_date=initial_date):

    daylight = pd.read_parquet(daylight_path)
    daylight['daylight_duration'] = (
        daylight['daylight_duration'] - daylight['daylight_duration'].min())
    daylight = daylight.groupby(by='date')[['daylight_duration']].mean() * -1

    weekend = pd.DataFrame({'ds': pd.date_range(
        start=initial_date, end=datetime.date(2051, 3, 31), freq='D')})
    weekend['sat_reg'] = weekend['ds'].apply(
        lambda x: daylight.loc[x, 'daylight_duration'] if x.weekday() == 5 else 0)
    weekend['sun_reg'] = weekend['ds'].apply(
        lambda x: daylight.loc[x, 'daylight_duration'] if x.weekday() == 6 else 0)

    temp = pd.read_parquet(h_weather_path).reset_index(drop=True)
    temp = temp[temp['date'] < temp['date'].max()]
    temp.columns = temp.columns.str.replace('gb_', '')
    temp = temp.rename(columns={'date': 'ds'})
    temp = temp.groupby('ds').agg(
        {'temperature': 'mean',
         'hdd': 'sum',
         'cdd': 'sum'}).reset_index()

    future_temp = pd.read_parquet(
        h_average_weahter_path).reset_index(drop=True)
    future_temp.columns = future_temp.columns.str.replace('gb_', '')
    future_temp = future_temp.rename(columns={'date': 'ds'})
    future_temp = future_temp.groupby('ds').agg(
        {'temperature': 'mean',
         'hdd': 'sum',
         'cdd': 'sum'}).reset_index()
    future_temp = future_temp[future_temp['ds'] > temp['ds'].max()]
    future_temp = pd.concat([temp, future_temp]).reset_index(drop=True)

    average_temp_all = pd.read_parquet(
        h_average_weahter_path).reset_index(drop=True)
    average_temp_all.columns = average_temp_all.columns.str.replace('gb_', '')
    average_temp_all = average_temp_all.rename(columns={'date': 'ds'})
    average_temp_all = average_temp_all.groupby('ds').agg(
        {'temperature': 'mean',
         'hdd': 'sum',
         'cdd': 'sum'}).reset_index()

    temp = temp.merge(weekend, how='left', on='ds')
    future_temp = future_temp.merge(weekend, how='left', on='ds')
    average_temp_all = average_temp_all.merge(weekend, how='left', on='ds')

    return (temp, future_temp, average_temp_all)


def population_transform(pop_dictionary):
    df_pop = pd.DataFrame(pop_dictionary)
    col = [x for x in df_pop.columns if x != 'year'][0]
    pop_ly = df_pop.copy()
    pop_ly['year'] = pop_ly['year'] + 1
    pop_ly = pop_ly.rename(columns={col: 'ly_'+col})
    df_pop = df_pop.merge(pop_ly)
    df_pop['delta'] = df_pop[col] - df_pop['ly_'+col]
    df_pop = df_pop[df_pop['year'] >= 2014]
    df_pop = df_pop.rename(columns={col: 'yearly_pop'})
    population = pd.DataFrame({'date': pd.date_range(
        start='2015-01-01', end='2045-01-01', freq='MS')})
    population['year'] = population['date'].dt.year
    population = pd.merge(population, df_pop, on='year')
    population = population.drop(columns='year')
    population[col] = population['ly_'+col] + \
        population['delta'] * ((population['date'].dt.month - 1)/12)
    population = population[['date', col]].rename(columns={'date': 'ds'})
    return population


def energy_cap_transform(cap_regressor):
    cap_regressor = pd.DataFrame(cap_regressor)
    col = [x for x in cap_regressor.columns if x != 'date'][0]
    cap_regressor['date'] = cap_regressor['date'].apply(
        lambda x: datetime.date(int(str(x)[:4]), int(str(x)[5:]), 1))
    cap_regressor['date'] = pd.to_datetime(cap_regressor['date'])
    cap_regressor = cap_regressor.rename(
        columns={'date': 'ds', col: 'energy_'+col})
    return cap_regressor


def gdp_transform(gdp_dic):
    gdp_final = pd.DataFrame(gdp_dic)
    gdp_final['ds'] = gdp_final['ds'].apply(
        lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:]), 1))
    gdp_final['ds'] = pd.to_datetime(gdp_final['ds'])
    return gdp_final


def make_monthly_regressors_df(category: Literal['Domestic', 'Non-domestic'], initial_date=initial_date, **kwargs):

    if 'gdp_dic' in kwargs.keys():
        gdp_dic = kwargs.get('gdp_dic')
    else:
        gdp_dic = default_gdp_dic

    if 'cap_regressor' in kwargs.keys():
        cap_regressor = kwargs.get('cap_regressor')
    else:
        cap_regressor = default_cap_regressor

    if 'pop_dictionary' in kwargs.keys():
        pop_dictionary = kwargs.get('pop_dictionary')
    else:
        pop_dictionary = default_pop_dictionary

    # Monthly Seasonality as a Binary + Accounting for leap years.
    df = pd.DataFrame({'ds': pd.date_range(
        start=initial_date, end='2050-03-01', freq='MS')})

    dic_int = {'is_jan': int,
               'is_feb': int,
               'is_leap': int,
               'is_mar': int,
               'is_apr': int,
               'is_may': int,
               'is_jun': int,
               'is_jul': int,
               'is_aug': int,
               'is_sep': int,
               'is_oct': int,
               'is_nov': int,
               'is_dec': int}
    month_binary = df.copy()
    month_binary['is_jan'] = month_binary['ds'].dt.month == 1
    month_binary['is_feb'] = month_binary['ds'].dt.month == 2
    month_binary['is_leap'] = month_binary['ds'].apply(
        lambda x: calendar.isleap(x.year) & (x.month == 2))
    month_binary['is_mar'] = month_binary['ds'].dt.month == 3
    month_binary['is_apr'] = month_binary['ds'].dt.month == 4
    month_binary['is_may'] = month_binary['ds'].dt.month == 5
    month_binary['is_jun'] = month_binary['ds'].dt.month == 6
    month_binary['is_jul'] = month_binary['ds'].dt.month == 7
    month_binary['is_aug'] = month_binary['ds'].dt.month == 8
    month_binary['is_sep'] = month_binary['ds'].dt.month == 9
    month_binary['is_oct'] = month_binary['ds'].dt.month == 10
    month_binary['is_nov'] = month_binary['ds'].dt.month == 11
    month_binary['is_dec'] = month_binary['ds'].dt.month == 12
    month_binary = month_binary.astype(dic_int)

    # Number of Weekdays in each month/year.
    weekdays = df.copy()
    weekdays['weekdays'] = weekdays['ds'].apply(lambda x: np.busday_count(datetime.date(x.year, x.month, x.day), datetime.date(
        x.year, x.month, x.day) + datetime.timedelta(calendar.monthrange(x.year, x.month)[1])))

    base_regressors = month_binary.merge(weekdays, how='left', on='ds')

    if category == 'Non-domestic':

        # Non-domestic statistical outliers.
        dic_int_2 = {'abnormal': int,
                     'covid_1': int,
                     'covid_2': int,
                     'covid_3': int}
        covid_binary = df.copy()
        covid_binary['abnormal'] = covid_binary['ds'].isin(
            ['2016-01-01', '2019-09-01'])
        covid_binary['covid_1'] = covid_binary['ds'].isin(
            ['2021-03-01', '2021-04-01'])
        covid_binary['covid_2'] = covid_binary['ds'].isin(
            ['2020-09-01', '2020-10-01', '2021-12-01'])
        covid_binary['covid_3'] = covid_binary['ds'].isin(
            ['2020-12-01', '2021-11-01'])
        covid_binary = covid_binary.astype(dic_int_2)

        # Non-domestic GDP regressor.
        gdp_final = gdp_transform(gdp_dic)
        monthly_regressors = gdp_final.merge(covid_binary, how='right', on='ds').merge(
            base_regressors, how='left', on='ds')

    else:

        # Domestic statistical outliers.
        dic_int_2 = {'covid_1': int,
                     'covid_2': int,
                     'covid_3': int,
                     'covid_4': int,
                     'covid_5': int}
        covid_binary = df.copy()
        covid_binary['covid_1'] = covid_binary['ds'].isin(
            ['2020-03-01', '2020-04-01', '2020-05-01'])
        covid_binary['covid_2'] = covid_binary['ds'].isin(
            ['2020-06-01', '2020-07-01', '2020-08-01'])
        covid_binary['covid_3'] = covid_binary['ds'].isin(
            ['2020-09-01', '2020-10-01', '2020-11-01'])
        covid_binary['covid_4'] = covid_binary['ds'].isin(
            ['2020-12-01', '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01', '2021-07-01'])
        covid_binary['covid_5'] = covid_binary['ds'].isin(
            ['2021-08-01', '2021-09-01'])
        covid_binary = covid_binary.astype(dic_int_2)

        # Domestic Population regressor.
        df_pop = population_transform(pop_dictionary)
        # Domestic EnergyCap regressor.
        cap_regressor = energy_cap_transform(cap_regressor)
        monthly_regressors = cap_regressor.merge(covid_binary, how='right', on='ds').merge(
            df_pop, how='left', on='ds').merge(base_regressors, how='left', on='ds')

    return monthly_regressors


def make_daily_seasonality_df(initial_date=initial_date):

    df = pd.DataFrame({'ds': pd.date_range(
        start=initial_date, end=datetime.date(2051, 3, 31), freq='D')})
    df['weekly_covid_addition'] = ((pd.to_datetime(df['ds']) >= pd.to_datetime(
        '2020-03-21')) & (pd.to_datetime(df['ds']) < pd.to_datetime('2021-03-21')))
    df['yearly_covid_addition'] = ((pd.to_datetime(df['ds']) >= pd.to_datetime(
        '2020-03-21')) & (pd.to_datetime(df['ds']) < pd.to_datetime('2022-02-20')))
    df['weekly_post_covid_addition'] = ((pd.to_datetime(df['ds']) >= pd.to_datetime(
        '2021-03-21')) & (pd.to_datetime(df['ds']) < pd.to_datetime('2022-02-20')))
    df['weekly_new_normal'] = (
        (pd.to_datetime(df['ds']) >= pd.to_datetime('2022-02-20')))
    df['yearly_post_covid_addition'] = (
        (pd.to_datetime(df['ds']) >= pd.to_datetime('2022-02-20')))

    return df


__all__ = ["make_daily_regressors_df",
           "make_monthly_regressors_df", "make_daily_seasonality_df"]
