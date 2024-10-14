import pandas as pd


def get_ensure_single_datetime_column(df: pd.DataFrame):

    datetime_columns = [
        col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    # Check if there is more than one datetime column
    if len(datetime_columns) > 1:
        raise IndexError(
            "There can only be 1 datetime column in the DataFrame.")
    # If there's exactly one datetime column, return it or perform other operations
    if len(datetime_columns) == 1:
        return datetime_columns[0]
    else:
        raise ValueError("No datetime column found in the DataFrame.")


def get_ensure_granularity(df: pd.DataFrame, date_col: str):

    if len(df[date_col].diff().dropna().unique()) != 1:
        raise IndexError("The DataFrame date steps aren't consistent")

    hours = df[date_col].diff().dropna().unique()[0].total_seconds()/3600

    if hours >= 600 and hours <= 770:
        granularity_symbol = 'month'
        granularity_nominal = 1
    if hours == 24:
        granularity_symbol = 'day'
        granularity_nominal = 1
    if hours == 1:
        granularity_symbol = 'hour'
        granularity_nominal = 1
    if hours < 1:
        granularity_symbol = 'minute'
        granularity_nominal = hours * 60

    return (granularity_symbol, granularity_nominal)


def stored_variables_names(variables):
    variable_names = [name for name, parameters in variables.items()]
    return variable_names


def ensure_variables_in_object(variable, variables_dict):

    variable_names = stored_variables_names(variables_dict)
    if variable not in variable_names:
        raise KeyError(
            f"Variable {variable} not in object's stored variables.")

    return


def ensure_variables_length(variables, variables_dict):

    lengths = []
    for variable in variables:
        lengths.append(variables_dict[variable]['scenarios_num'])

    if len(set(lengths)) != 1:
        raise ValueError(
            "The variables that want to be paired do not have the exact same number of scenarios.")
