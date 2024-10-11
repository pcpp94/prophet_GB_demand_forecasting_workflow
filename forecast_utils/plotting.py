import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from pandas.plotting import deregister_matplotlib_converters
import seaborn as sns
import pandas as pd
from prophet.plot import *

from .default_variables import *
deregister_matplotlib_converters()


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


__all__ = ["plot_regressors_linearity", "plot_regressors", "plot_inputs_in_time",
           "plot_noise_ts", "plot_error_hist", "plot_forecast_changepoints", "plot_base_components"]
