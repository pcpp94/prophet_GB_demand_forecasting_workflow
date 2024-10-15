import pandas as pd
from prophet import Prophet
from collections import defaultdict
from typing import List
from .utils import get_ensure_single_datetime_column, get_ensure_granularity, ensure_variables_in_object, ensure_variables_length


class Scenarios_Client:
    def __init__(self, granular_model: Prophet, monthly_model: Prophet):
        self.granular_model = granular_model
        self.monthly_model = monthly_model
        self.paired_variables = None

        # Getting the granularity of the "Granular Model": Could be Hourly, 30min, 15min...
        self.granular_model_granularity_symbol, self.granular_model_granularity_nominal = get_ensure_granularity(
            granular_model.history, date_col='ds')
        self.granular_granularity = str(
            self.granular_model_granularity_nominal) + self.granular_model_granularity_symbol

        # Making sure a monthly model was given:
        self.monthly_model_granularity_symbol, self.monthly_model_granularity_nominal = get_ensure_granularity(
            monthly_model.history, date_col='ds')
        self.monthly_granularity = str(
            self.monthly_model_granularity_nominal) + self.monthly_model_granularity_symbol
        if self.monthly_granularity != '1month':
            raise TypeError("A monthly model is needed as an input")

        # Variables' Dictionary.
        self.variables = defaultdict(lambda: defaultdict(dict))

    def add_variable(self, variable_df: pd.DataFrame, variable_name: str):

        self.date_col = get_ensure_single_datetime_column(variable_df)
        granularity_symbol, granularity_nominal = get_ensure_granularity(
            variable_df, self.date)
        granularity = str(granularity_nominal) + granularity_symbol

        # Checking that the granularity of the variables matches the granularity of the models:
        if granularity not in [self.granular_granularity, '1month']:
            raise TypeError(
                "Granularity of variable doesn't match models' granularities.")

        # Number of scenarios
        scenarios = variable_df.select_dtypes("number").columns

        # Storing the parameters of the variable
        self.variables[variable_name]['granularity_symbol'] = granularity_symbol
        self.variables[variable_name]['granularity_nominal'] = granularity_nominal
        self.variables[variable_name]['granularity'] = granularity
        self.variables[variable_name]['scenarios_num'] = len(scenarios)
        self.variables[variable_name]['table'] = variable_df

        return f"Added {self.variables}"

    def variables_pairing(self, variables: List[str]):

        for variable in variables:
            ensure_variables_in_object(variable, self.variables)
        ensure_variables_length(variables, self.variables)
        self.paired_variables = variables

        return f"Paired variables: {variables}"

    def create_scenarios(self):
        """
        Variables that are not paired will be cross-merged.
        Variables that are paired need to have the same length as these will be merged 1-1.
        """

        # Create the scenarios as a DataFrame which has columns: scenario_num, scenario, variable1, variable2, ...
        # scenario_num would be: 1, 2, 3...
        # scenario would be the name of the columns in the variable_df DataFrames.
        # variables: the values.

        if len(self.variables) == 0:
            raise ValueError("No variables have been added to the object.")

        df_aux = self.variables['temperature']['table'].copy()


# Example:
# import pandas as pd
# from itertools import product
# import os

# df1 = pd.DataFrame({"date": pd.date_range(start="2015-01-01", end="2024-01-10", freq='D')})
# df2 = pd.DataFrame({"date": pd.date_range(start="2015-01-01", end="2024-01-10", freq='D')})

# for x in range(20):
#     df1['temp_'+str(x+1)] = x ** 2

# for x in range(20):
#     df2['cdd_'+str(x+1)] = x ** 2.2

# paired:
# list(zip(var1,var2))

# cross:
# list(product(var1,var2))
