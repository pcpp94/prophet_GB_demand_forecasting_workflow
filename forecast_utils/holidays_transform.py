import pandas as pd
from prophet.make_holidays import make_holidays_df


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


def tailored_holidays_gb(initial_year=2015, final_year=2050):
    """
    Make dataframe of GB  holidays for given years,
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

    # Creating Holidays df
    holidays_dic = {
        1: 'dow12',
        2: 'dow12',
        3: 'dow34',
        4: 'dow34',
        5: 'dow5',
        6: 'dow6',
        7: 'dow7'}
    year_list = list(range(2015, 2024+1))
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
    holidays = pd.concat([holidays, lockdowns]).reset_index(drop=True)
    holidays = pd.concat([holidays, extra_holidays]).reset_index(drop=True)
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


__all__ = ["holiday_names_func",
           "xmas_ny_func", "holidays_by_date", "tailored_holidays_gb"]
