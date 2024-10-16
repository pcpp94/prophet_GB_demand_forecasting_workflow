import pandas as pd
from prophet import Prophet
from collections import defaultdict
from typing import List
from itertools import product
from .utils import get_ensure_single_datetime_column, get_ensure_granularity, ensure_variables_in_object, ensure_variables_length


class Scenarios_Client:
    def __init__(self, granular_model: Prophet, monthly_model: Prophet):
        self.granular_model = granular_model
        self.monthly_model = monthly_model
        self.paired_variables = []
        self.scenarios_dfs = []
        self.variables_granularities = set()

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

        self.added_variables = []
        # Variables' Dictionary.
        self.variables = defaultdict(lambda: defaultdict(dict))

    def add_variable(self, variable_df: pd.DataFrame, variable_name: str):

        date_col = get_ensure_single_datetime_column(variable_df)
        granularity_symbol, granularity_nominal = get_ensure_granularity(
            variable_df, date_col=date_col)
        granularity = str(granularity_nominal) + granularity_symbol
        # Ensuring all dataframes have "ds" as the date column.
        variable_df = variable_df.rename(columns={date_col: 'ds'})

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
        self.variables_granularities.add(granularity)
        self.variables[variable_name]['scenarios_num'] = len(scenarios)
        self.variables[variable_name]['table'] = variable_df

        self.added_variables.append(variable_name)

        return f"Added {variable_name}"

    def variables_pairing(self, variables: List[str]):

        for variable in variables:
            ensure_variables_in_object(variable, self.variables)
        ensure_variables_length(variables, self.variables)
        self.paired_variables = variables

        return f"Paired variables: {variables}"

    def create_scenarios_list(self):
        """
        Variables that are not paired will be cross-merged.
        Variables that are paired need to have the same length as these will be merged 1-1.
        """

        if len(self.variables) == 0:
            raise ValueError("No variables have been added to the object.")

        aux_dict = {}
        paired = []
        cross = []
        names = []

        for var in self.variables.keys():
            aux_dict[var] = self.variables[var]['table'].select_dtypes(
                'number').columns.tolist()

        if len(self.paired_variables) > 1:
            for var in self.paired_variables:
                paired.append(aux_dict[var])
                names.append(var)
            paired_scenarios_list = list(zip(*paired))

        for var in [x for x in self.variables.keys() if x not in names]:
            cross.append(aux_dict[var])
            names.append(var)

        if len(paired) == 0:
            final_scenarios_list = list(product(*cross))
        else:
            final_scenarios_list = list(product(paired_scenarios_list, *cross))

        final_scenarios_list = [(*nested[0], *nested[1:])
                                for nested in final_scenarios_list]

        return (final_scenarios_list, names)

    def create_scenarios(self):
        """
        The output is a list including Dictionaries of Pandas DataFrames.
        Each Dictionary has 2 DataFrames for a monthly-granularity DataFrame. and a "granular"-granularity DataFrame.
        """

        if len(self.variables) == 0:
            raise ValueError("No variables have been added to the object.")

        final_scenarios_list, variables_order = self.create_scenarios_list()

        for scenario in final_scenarios_list:

            granularity_checks = []
            df_dict = defaultdict(lambda: defaultdict(dict))

            for var, name in zip(scenario, variables_order):
                aux = self.variables[name]['table'][['ds', var]].copy()
                granularity = self.variables[name]['granularity']
                granularity_checks.append(granularity)
                if len(df_dict[granularity]) == 0:
                    df_dict[granularity] = aux.copy()
                else:
                    df_dict[granularity] = df_dict[granularity].merge(
                        aux, how='left', on=['ds'])
                if len(set(granularity_checks)) > 2:
                    raise ValueError(
                        "The variables need to be of granularity: 1month and an extra one")

            for gran in self.variables_granularities:
                df_dict[gran] = df_dict[gran].dropna()
            self.scenarios_dfs.append(df_dict)

        return self.scenarios_dfs
