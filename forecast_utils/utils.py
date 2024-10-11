import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from pandas.plotting import deregister_matplotlib_converters
import seaborn as sns
import datetime
import pandas as pd
from prophet.make_holidays import make_holidays_df
from prophet.plot import *


yearly_regressors = ['is_apr', 'is_aug', 'is_dec', 'is_feb', 'is_jan', 'is_jul',
                     'is_jun', 'is_leap', 'is_mar', 'is_may', 'is_nov', 'is_oct', 'is_sep', 'weekdays']
default_monthly_regressors = ['gdp', 'energy_cap', 'population']
weather_regressors = ['temperature', 'hdd',
                      'cdd', 'snow_depth', 'precipitation_mm']


deregister_matplotlib_converters()


def reducing(x, y):
    if len(str(x)) == 19:
        return str(x)[0:10] + ',' + str(y)[0:10]
    else:
        return str(x) + ',' + str(y)[0:10]


def holiday_names_func(holidays):
    holiday_names = pd.DataFrame({'holidays': holidays['holiday'].unique()})
    return holiday_names


def xmas_ny_func(holiday_names):
    xmas_ny = holiday_names[holiday_names['holidays'].str.slice(
        stop=4) == 'xmas']
    xmas_ny = xmas_ny['holidays'].str.split('_', expand=True)
    xmas_ny['group'] = xmas_ny[0] + '_' + \
        xmas_ny[1] + '_' + xmas_ny[2] + '_' + xmas_ny[3]
    xmas_ny['individual'] = xmas_ny[0] + '_' + xmas_ny[1] + \
        '_' + xmas_ny[2] + '_' + xmas_ny[3] + '_' + xmas_ny[4]
    xmas_ny = xmas_ny.drop(columns=[0, 1, 2, 3, 4]).reset_index(drop=True)
    return xmas_ny


def holidays_by_date(holidays, holiday_names):
    holidays_df = holidays[['ds', 'holiday']].rename(
        columns={'holiday': 'holiday_name'})
    holidays_df['holiday_name'] = holidays_df['holiday_name'].replace(dict(zip(xmas_ny_func(holiday_names).to_dict(
        orient='series')['individual'], xmas_ny_func(holiday_names).to_dict(orient='series')['group'])))
    return holidays_df


def hyper_params_df(trials):

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
    params['model_mdape'] = trials.best_trial['result']['metrics']['mape']
    params['training_datetime'] = trials.best_trial['result']['training_datetime']

    return params


def full_forecast_df(model):

    if model.granularity == 'Daily':
        seasonalities = [i for i in model.seasonalities]
        regressors = [i for i in model.extra_regressors]
        for seasonality in seasonalities:
            model.future = model.future.rename(
                columns={seasonality: f"flag_{seasonality}"})
        for regressor in regressors:
            model.future = model.future.rename(
                columns={regressor: f"nominal_{regressor}"})
        if model.holidays is None:
            columns_to_use = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper',
                              'trend_lower', 'trend_upper', 'multiplicative_terms', 'additive_terms']
        else:
            columns_to_use = ['ds', 'trend', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower',
                              'trend_upper', 'holidays', 'multiplicative_terms', 'additive_terms']
        holiday_columns = list(model.holiday_names['holidays'])
        columns_to_use.extend(seasonalities + regressors)
        columns_to_use.extend(holiday_columns)
        raw_forecast = model.forecast[columns_to_use]
        full_forecast = model.history[['ds', 'y']].merge(
            model.future, how='right', on='ds').merge(raw_forecast, how='inner', on='ds')
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
                                                            list(model.xmas_ny['individual'])].sum(axis=1)
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
            columns=list(model.xmas_ny['individual']))
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
            model.future = model.future.rename(
                columns={regressor: f"nominal_{regressor}"})
        columns_to_use.extend(regressors)
        raw_forecast = model.forecast[columns_to_use]
        full_forecast = model.history[['ds', 'y']].merge(
            model.future, how='right', on='ds').merge(raw_forecast, how='inner', on='ds')
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


def reduced_forecast_df(model):

    if model.granularity == 'Monthly':
        reduced_forecast = model.full_forecast.copy()
        externalities_list = [
            i for i in model.extra_regressors if i not in yearly_regressors+default_monthly_regressors]
        drop_list = ["nominal_"+regressor for regressor in externalities_list] + ["nominal_" +
                                                                                  regressor for regressor in yearly_regressors] + externalities_list + yearly_regressors
        drop_list.remove('nominal_weekdays')
        drop_list.remove('weekdays')
        drop_list.remove('is_leap')
        reduced_forecast = reduced_forecast.drop(columns=drop_list)
    else:
        holidays_date = model.holidays_date
        reduced_forecast = model.full_forecast.copy()
        cols = reduced_forecast.columns
        cols = cols[~cols.isin(list(model.holiday_names['holidays']))]
        reduced_forecast = reduced_forecast[cols]
        reduced_forecast = reduced_forecast.merge(
            holidays_date, how='left', on='ds')

    return reduced_forecast


def plot_regressors_linearity(model):

    if model.full_forecast is None:
        raise Exception("Full Forecast method must be run first")

    regressors = default_monthly_regressors + weather_regressors
    components = []
    df = model.full_forecast.dropna()

    for regressor in regressors:
        if regressor in list(model.extra_regressors.keys()):
            components.append(regressor)

    if 'temperature' in components:
        components.append('total_temperature')

    npanel = len(components)

    figsize = (9, 5 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    for ax, component in zip(axes, components):
        if component == 'total_temperature':
            ax.scatter(df['nominal_temperature'],
                       df['temperature']+df['hdd']+df['cdd'])
            ax.set_xlabel('nominal_temperature')
            ax.set_ylabel(component)
        else:
            ax.scatter(df['nominal_' + component], df[component])
            ax.set_xlabel('nominal_' + component)
            ax.set_ylabel(component)

    return fig


def plot_regressors(model):

    if model.full_forecast is None:
        raise Exception("Full Forecast method must be run first")

    components = list(model.extra_regressors)
    df = model.full_forecast.set_index('ds')
    seasonality_ax = df.dropna()[components].plot(
        figsize=(15, 10), title='Regressors', alpha=0.5)
    seasonality_fig = seasonality_ax.get_figure()

    return seasonality_fig


def plot_inputs_in_time(model):

    if model.full_forecast is None:
        raise Exception("Full Forecast method must be run first")

    components = list(model.seasonalities) + \
        list(model.extra_regressors)
    df = model.full_forecast.set_index('ds')
    seasonality_ax = df.dropna()[components].plot(subplots=True, figsize=(
        15, 10), title='Model Components [Seasonality, Regressors]')
    seasonality_fig = seasonality_ax[0].get_figure()

    return seasonality_fig


def plot_noise_ts(model):

    if model.full_forecast is None:
        raise Exception("Full Forecast method must be run first")

    error_ax = model.full_forecast[model.full_forecast['ds'] >= '2015-01-01'].dropna().set_index('ds')['residual'].plot(figsize=(
        15, 5), title='Errors | Noise | Residuals', xlim=([model.history['ds'].min(), model.history['ds'].max()]))
    error_fig = error_ax.get_figure()

    return error_fig


def plot_error_hist(model):

    if model.full_forecast is None:
        raise Exception("Full Forecast method must be run first")

    sns.light_palette('seagreen', as_cmap=True)
    f, ax = plt.subplots(figsize=(8, 8))
    sns.histplot(
        model.full_forecast['error_percentage'].dropna()*100, ax=ax, color='blue')
    ax.grid(ls=':')
    ax.set_xlabel('Residuals', fontsize=15)
    ax.set_ylabel("Frequency", fontsize=15)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(ls=':')
    ax.axvline(0, color='0.4')
    ax.set_title('Residuals Distribution [Backcast]', fontsize=17)
    ax.text(0.05, 0.85, "MDAPE = {:4.2%}\nMAPE = {:4.2%}".format(
        model.full_forecast['error_percentage'].dropna().abs().median(), model.full_forecast['error_percentage'].dropna().abs().mean()), fontsize=14, transform=ax.transAxes)

    return f


def plot_forecast_changepoints(model):

    fig = model.plot(model.forecast)
    a = add_changepoints_to_plot(
        fig.gca(), model, model.forecast)
    return fig


def plot_base_components(model, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None):

    # Identify components to be plotted
    yearly_regressors_ = [i for i in yearly_regressors if i in list(
        model.extra_regressors.keys())]
    default_monthly_regressors_ = [
        i for i in default_monthly_regressors if i in list(model.extra_regressors.keys())]
    weather_regressors_ = [i for i in weather_regressors if i in list(
        model.extra_regressors.keys())]

    components = ['trend']
    if model.train_holiday_names is not None and 'holidays' in model.forecast:
        components.append('holidays')
    # Plot weekly seasonality, if present
    if 'weekly' in model.seasonalities and 'weekly' in model.forecast:
        components.append('weekly')
    # Yearly if present
    if 'yearly' in model.seasonalities and 'yearly' in model.forecast:
        components.append('yearly')

    # Regressors split by type depending on the model.
    regressors = dict(model.extra_regressors.items())
    seasonality_regressors_list = []
    socio_economic_monthly_regressors_list = []
    weather_regressors_list = []
    externality_regressors_list = []

    for yearly in yearly_regressors_:
        if regressors[yearly] and yearly in model.forecast:
            seasonality_regressors_list.append(yearly)
            if 'extra_regressors_seasonality' not in components:
                components.append('extra_regressors_seasonality')
    for socio_economic in default_monthly_regressors_:
        if regressors[socio_economic] and socio_economic in model.forecast:
            socio_economic_monthly_regressors_list.append(socio_economic)
            if 'extra_regressors_socio_economic' not in components:
                components.append('extra_regressors_socio_economic')
    for weather in weather_regressors_:
        if regressors[weather] and weather in model.forecast:
            weather_regressors_list.append(weather)
            if 'extra_regressors_weather' not in components:
                components.append('extra_regressors_weather')
    for externalities in [i for i in model.extra_regressors if i not in yearly_regressors_+default_monthly_regressors_+weather_regressors_]:
        if regressors[externalities] and externalities in model.forecast:
            externality_regressors_list.append(externalities)
            if 'extra_regressors_externalities' not in components:
                components.append('extra_regressors_externalities')
    regressors_dict = dict(zip(['extra_regressors_seasonality', 'extra_regressors_socio_economic', 'extra_regressors_weather', 'extra_regressors_externalities'],
                               [seasonality_regressors_list, socio_economic_monthly_regressors_list,
                                   weather_regressors_list, externality_regressors_list]
                               ))

    npanel = len(components)

    for cat in regressors_dict.keys():
        if len(regressors_dict[cat]) > 0:
            model.forecast[cat] = model.forecast[regressors_dict[cat]].sum(
                axis=1)
            model.forecast[cat +
                           '_lower'] = model.forecast[regressors_dict[cat]].sum(axis=1)
            model.forecast[cat +
                           '_upper'] = model.forecast[regressors_dict[cat]].sum(axis=1)

    figsize = figsize if figsize else (9, 3 * npanel)
    fig, axes = plt.subplots(npanel, 1, facecolor='w', figsize=figsize)

    if npanel == 1:
        axes = [axes]

    multiplicative_axes = []

    dt = model.history['ds'].diff()
    min_dt = dt.iloc[dt.values.nonzero()[0]].min()

    for ax, plot_name in zip(axes, components):
        if plot_name == 'trend':
            plot_forecast_component(
                m=model, fcst=model.forecast, name='trend', ax=ax, uncertainty=uncertainty,
                plot_cap=plot_cap,
            )
        elif plot_name in model.seasonalities:
            if (
                (plot_name ==
                 'weekly' or model.seasonalities[plot_name]['period'] == 7)
                and (min_dt == pd.Timedelta(days=1))
            ):
                plot_weekly(
                    m=model, name=plot_name, ax=ax, uncertainty=uncertainty, weekly_start=weekly_start
                )
            elif plot_name == 'yearly' or model.seasonalities[plot_name]['period'] == 365.25:
                plot_yearly(
                    m=model, name=plot_name, ax=ax, uncertainty=uncertainty, yearly_start=yearly_start
                )
            else:
                plot_seasonality(
                    m=model, name=plot_name, ax=ax, uncertainty=uncertainty,
                )
        elif plot_name in ['holidays', 'extra_regressors_seasonality', 'extra_regressors_socio_economic', 'extra_regressors_weather', 'extra_regressors_externalities']:
            plot_forecast_component(m=model, fcst=model.forecast, name=plot_name,
                                    ax=ax, uncertainty=uncertainty, plot_cap=False)
        if plot_name != 'trend':
            multiplicative_axes.append(ax)

    fig.tight_layout()

    # Reset multiplicative axes labels after tight_layout adjustment
    for ax in multiplicative_axes:
        ax = set_y_as_percent(ax)

    return fig


def tailored_holidays_gb(initial_year=2015, final_year=2050):
    """
    Make dataframe of GB holidays for given years,
    adding the day-of-week variables to catch more signals during
    Xmas and NYs time (Very day-of-week dependent).
    To correct for COVID-19 lockdowns and the Beast of the East, some one-time
    holidays were used.

    Parameters
    ----------
    initial_year: beginning year.
    final_year: final year.

    Returns
    -------
    Dataframe with 'ds', 'holiday', 'lower_window', 'upper_window'
    which can directly feed to 'holidays' params in Prophet
    """

    holidays_dic = {
        1: 'dow12',
        2: 'dow12',
        3: 'dow34',
        4: 'dow34',
        5: 'dow5',
        6: 'dow6',
        7: 'dow7'}

    year_list = list(range(initial_year, final_year+1))
    holidays = make_holidays_df(year_list=year_list, country='UK')

    # Dropping Norther Ireland's holidays & Christmas Holidays as these will be added depending on the day-of-week
    northern_ireland = ["St. Patrick's Day [Northern Ireland]",
                        "Battle of the Boyne [Northern Ireland]", "St. Patrick's Day [Northern Ireland] (Observed)"]
    holidays = holidays[~holidays['holiday'].isin(northern_ireland)]
    drop_xmas_ny = [
        'Boxing Day', 'Boxing Day (Observed)', 'Christmas Day', 'New Year Holiday [Scotland]', "New Year's Day"]
    holidays = holidays[~holidays['holiday'].isin(drop_xmas_ny)]
    holidays['lower_window'] = -1
    holidays['upper_window'] = 1

    # Lockdowns + Beast of the East as one-off holiday
    lockdowns = pd.DataFrame([
        {'holiday': 'beast_of_the_east', 'ds': '2018-02-24',
            'lower_window': 0, 'ds_upper': '2018-04-19'},
        {'holiday': 'lockdown_1', 'ds': '2020-03-21',
            'lower_window': 0, 'ds_upper': '2020-06-30'},
        {'holiday': 'lockdown_2', 'ds': '2020-10-25',
            'lower_window': 0, 'ds_upper': '2020-12-15'},
        {'holiday': 'lockdown_3', 'ds': '2021-02-15', 'lower_window': 0, 'ds_upper': '2021-03-12'}])
    for t_col in ['ds', 'ds_upper']:
        lockdowns[t_col] = pd.to_datetime(lockdowns[t_col])
    lockdowns['upper_window'] = (
        lockdowns['ds_upper'] - lockdowns['ds']).dt.days
    lockdowns = lockdowns.drop(columns='ds_upper')

    # Christmas Holidays depending on the day-of-week
    add_xmas_ny = ["12_22", "12_23", "12_24", "12_25", "12_26",
                   "12_27", "12_28", "12_29", "12_30", "12_31", "1_1", "1_2"]
    extra_holidays = pd.DataFrame(
        {'ds': pd.date_range(start='2015-01-01', end='2050-12-31')})
    extra_holidays['month_day'] = extra_holidays['ds'].dt.month.astype(
        str) + '_' + extra_holidays['ds'].dt.day.astype(str)
    extra_holidays = extra_holidays[extra_holidays['month_day'].isin(
        add_xmas_ny)]
    extra_holidays['group'] = extra_holidays['ds'].dt.dayofweek + 1
    extra_holidays['group'] = extra_holidays['group'].map(holidays_dic)
    extra_holidays.loc[:, 'holiday'] = 'xmas_ny_' + extra_holidays.loc[:,
                                                                       'month_day'] + '_' + extra_holidays.loc[:, 'group']
    extra_holidays = extra_holidays.drop(columns=['month_day', 'group'])
    extra_holidays['lower_window'] = 0
    extra_holidays['upper_window'] = 0

    # Merging the DataFrames
    holidays = holidays.append(lockdowns).reset_index(drop=True)
    holidays = holidays.append(extra_holidays).reset_index(drop=True)
    holidays = holidays.sort_values('ds')
    holidays['holiday'] = holidays['holiday'].str.lower()
    holidays['holiday'] = holidays['holiday'].str.replace(
        " ", "_", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace("'", "", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace(".", "", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace(
        "/", "_", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace("(", "", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace(")", "", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace("[", "", regex=False)
    holidays['holiday'] = holidays['holiday'].str.replace("]", "", regex=False)

    return holidays
