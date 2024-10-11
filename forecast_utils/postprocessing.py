import pandas as pd
import datetime
from .default_variables import *
from .holidays_transform import *


def hyper_params_df(trials):
    """
    Gets values retrieved by Trials() object from HyperOpt and cleans them.

    Parameters
    ----------
    trials: hyperopt Trials() object.

    Returns
    -------
    Dataframe with the HyperOpt trials variables and the Prophet model's hyperparameters.
    """
    params = pd.DataFrame.from_dict([trials.argmin])
    params['cross_val_mape'] = trials.best_trial['result']['loss']
    params['cutoff_points'] = trials.best_trial['result']['cutoff_points']
    params['horizon_days'] = trials.best_trial['result']['horizon_days']
    params['hopt_algorithm'] = trials.best_trial['result']['hopt_algorithm']
    params['max_iters'] = trials.best_trial['result']['max_iters']
    params['random_state'] = trials.best_trial['result']['random_state']
    params['model_sector'] = trials.best_trial['result']['category']
    params['model_granularity'] = trials.best_trial['result']['granularity']
    params['model_mape'] = trials.best_trial['result']['metrics']['mape']
    params['model_mdape'] = trials.best_trial['result']['metrics']['mdape']
    params['training_datetime'] = trials.best_trial['result']['training_datetime']

    return params


def full_forecast_df(model, future, forecast):
    """
    Makes an organized forecast DataFrame including input values that are valuable.
    The Prophet model object needs an attribute Prophet().granularity -> which can be Daily or Monthly.

    Parameters
    ----------
    model: Prophet model object with "granularity" as an attribute.

    Returns
    -------
    Dataframe with a Full Forecast.
    """

    if hasattr(model, 'granularity'):
        pass
        # Proceed with using obj.granularity
    else:
        raise AttributeError(
            "The passed object lacks a 'granularity' attribute.")

    if model.granularity == 'D':
        holiday_names = holiday_names_func(model.holidays)
        xmas_ny = xmas_ny_func(holiday_names)
        seasonalities = [i for i in model.seasonalities]
        regressors = [i for i in model.extra_regressors]
        for seasonality in seasonalities:
            future = future.rename(
                columns={seasonality: f"flag_{seasonality}"})
        for regressor in regressors:
            future = future.rename(
                columns={regressor: f"nominal_{regressor}"})
        if model.holidays is None:
            columns_to_use = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper',
                              'trend_lower', 'trend_upper', 'multiplicative_terms', 'additive_terms']
        else:
            columns_to_use = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower',
                              'trend_upper', 'holidays', 'multiplicative_terms', 'additive_terms']
        holiday_columns = list(holiday_names['holidays'])
        columns_to_use.extend(seasonalities + regressors)
        columns_to_use.extend(holiday_columns)
        raw_forecast = forecast[columns_to_use]
        full_forecast = model.history[['ds', 'y']].merge(
            future, how='right', on='ds').merge(raw_forecast, how='inner', on='ds')
        full_forecast.columns = full_forecast.columns.str.lower()
        full_forecast.columns = full_forecast.columns.str.replace(
            " ", "_", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "'", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            ".", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "/", "_", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "(", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            ")", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "[", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "]", "", regex=False)
        full_forecast.loc[:, 'xmas_ny'] = full_forecast.loc[:,
                                                            list(xmas_ny['individual'])].sum(axis=1)
        full_forecast['residual'] = full_forecast['yhat'] - full_forecast['y']
        full_forecast['error_percentage'] = full_forecast['residual'] / \
            full_forecast['y']
        residuals = full_forecast['error_percentage'].dropna()
        full_forecast.loc[:, 'total_yearly'] = full_forecast.loc[:, [
            i for i in model.seasonalities if i.startswith('yearly')]].sum(axis=1)
        full_forecast.loc[:, 'total_weekly'] = full_forecast.loc[:, [i for i in model.seasonalities if i.startswith(
            'weekly')] + [i for i in model.extra_regressors if i.startswith('sat') or i.startswith('sun')]].sum(axis=1)
        full_forecast.loc[:, 'weekly_weekend'] = full_forecast.loc[:, [
            i for i in model.extra_regressors if i.startswith('sat') or i.startswith('sun')]].sum(axis=1)
        full_forecast = full_forecast.drop(
            columns=list(xmas_ny['individual']))
        full_forecast['yhat_detrended'] = full_forecast['yhat_upper'] / \
            full_forecast['trend_upper']
        full_forecast['date_month'] = full_forecast['ds'].dt.to_period(
            'M')
        full_forecast['daily_weighting'] = full_forecast.groupby(
            by='date_month')['yhat_detrended'].transform(lambda x: x/x.sum())
        full_forecast = full_forecast.drop(columns='date_month')
        full_forecast['training_datetime'] = datetime.datetime.today()

    else:
        regressors = [i for i in model.extra_regressors]
        columns_to_use = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper',
                          'trend_lower', 'trend_upper', 'multiplicative_terms', 'additive_terms']
        for regressor in regressors:
            future = future.rename(
                columns={regressor: f"nominal_{regressor}"})
        columns_to_use.extend(regressors)
        raw_forecast = forecast[columns_to_use]
        full_forecast = model.history[['ds', 'y']].merge(
            future, how='right', on='ds').merge(raw_forecast, how='inner', on='ds')
        full_forecast['yearly'] = full_forecast[yearly_regressors].sum(axis=1)
        full_forecast['externalities'] = full_forecast[[
            i for i in model.extra_regressors if i not in yearly_regressors+default_monthly_regressors]].sum(axis=1)
        full_forecast.columns = full_forecast.columns.str.lower()
        full_forecast.columns = full_forecast.columns.str.replace(
            " ", "_", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "'", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            ".", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "/", "_", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "(", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            ")", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "[", "", regex=False)
        full_forecast.columns = full_forecast.columns.str.replace(
            "]", "", regex=False)
        full_forecast['residual'] = full_forecast['yhat'] - full_forecast['y']
        full_forecast['error_percentage'] = full_forecast['residual'] / \
            full_forecast['y']
        full_forecast['yhat_detrended'] = full_forecast['yhat_upper'] / \
            full_forecast['trend_upper']
        full_forecast['training_datetime'] = datetime.datetime.today()

    return full_forecast


def reduced_forecast_df(model, full_forecast):
    """
    Creates a reduced forecast by merging some individual variables, like the holiday effect from each holiday into a single columns

    Parameters
    ----------
    model: Prophet model object with "granularity" as an attribute.

    Returns
    -------
    Dataframe with a Reduced Forecast.
    """

    if model.granularity == 'M':
        reduced_forecast = full_forecast.copy()
        externalities_list = [
            i for i in model.extra_regressors if i not in yearly_regressors+default_monthly_regressors]
        drop_list = ["nominal_"+regressor for regressor in externalities_list] + ["nominal_" +
                                                                                  regressor for regressor in yearly_regressors] + externalities_list + yearly_regressors
        drop_list.remove('nominal_weekdays')
        drop_list.remove('weekdays')
        drop_list.remove('is_leap')
        reduced_forecast = reduced_forecast.drop(columns=drop_list)
    else:
        holiday_names = holiday_names_func(model.holidays)
        holidays_date = holidays_by_date(model.holidays, holiday_names)
        reduced_forecast = full_forecast.copy()
        cols = reduced_forecast.columns
        cols = cols[~cols.isin(list(holiday_names['holidays']))]
        reduced_forecast = reduced_forecast[cols]
        reduced_forecast = reduced_forecast.merge(
            holidays_date, how='left', on='ds')

    return reduced_forecast


__all__ = ["hyper_params_df", "full_forecast_df", "reduced_forecast_df"]
