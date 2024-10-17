from .utils import get_ensure_single_datetime_column, get_ensure_granularity, ensure_variables_in_object, ensure_variables_length
from .config import scenario_outputs
import pandas as pd
import sys
import os
import datetime
from prophet import Prophet
import mlflow
from mlflow.tracking import MlflowClient
from collections import defaultdict
from typing import List
from itertools import product

levels = 1
sys.path.append(os.path.abspath(os.path.join(".", "../"*levels)))
import forecast_utils as utils

class Scenarios_Client:
    def __init__(self, granular_model: mlflow.pyfunc.PyFuncModel, monthly_model: mlflow.pyfunc.PyFuncModel, mlflow_uri: str):
        self.paired_variables = []
        self.scenarios_dfs = []
        self.variables_granularities = set()
        self.create_scenarios_check = False

        # Client for forecast retrieval:
        mlflow.set_tracking_uri(mlflow_uri)
        client = MlflowClient()

        # Monthly Model forecast retrieval and Get Prophet Object:
        monthly_run_id = monthly_model.metadata.run_id
        self.monthly_forecast = pd.read_parquet(client.download_artifacts(
            monthly_run_id, "forecast_outputs/forecast_full_df.parquet"))
        self.monthly_model = monthly_model.get_raw_model()

        # Daily Model forecast retrieval and Get Prophet Object:
        daily_run_id = granular_model.metadata.run_id
        self.granular_forecast = pd.read_parquet(client.download_artifacts(
            daily_run_id, "forecast_outputs/forecast_full_df.parquet"))
        self.granular_model = granular_model.get_raw_model()

        # Getting the granularity of the "Granular Model": Could be Hourly, 30min, 15min...
        self.granular_model_granularity_symbol, self.granular_model_granularity_nominal = get_ensure_granularity(
            self.granular_model.history, date_col='ds')
        self.granular_granularity = str(
            self.granular_model_granularity_nominal) + self.granular_model_granularity_symbol

        # Making sure a monthly model was given:
        self.monthly_model_granularity_symbol, self.monthly_model_granularity_nominal = get_ensure_granularity(
            self.monthly_model.history, date_col='ds')
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

        self.create_scenarios_check = True
        return self.scenarios_dfs

    def run_ony_granular_scenario(self, scenario_number):

        if self.create_scenarios_check == False:
            raise ValueError(
                "Scenarios have not been created, run method 'create_scenarios()")

        granularity_one_digit = [x for x in list(
            self.variables_granularities) if x != '1month'][0][1:].upper()[0]
        category = self.granular_model.history['sector'].unique()[0]
        # Putting the Granular scenario variables into the "Future DF"

        # Getting the variables' ordered names:
        granular_scenario_variables_ordered_list = []
        granular_granularity = [x for x in list(
            self.variables_granularities) if x != '1month'][0]
        for var in self.variables.keys():
            if self.variables[var]["granularity"] == granular_granularity:
                granular_scenario_variables_ordered_list.append(var)

        # Getting the scenario NUMBER 0 regressors and changing the col names to the ordered variable names (from temperature_1 to temperature for example)
        scenarios_df = self.scenarios_dfs
        gran_regs_df = scenarios_df[scenario_number][granular_granularity].copy(
        )
        fc_date = gran_regs_df['ds'].max()
        scen_cols = [x for x in gran_regs_df.columns if x != 'ds']
        gran_regs_df = gran_regs_df.rename(columns=dict(
            zip(scen_cols, granular_scenario_variables_ordered_list)))

        # Creating the future_df and replacing the "default variables" for the new values from the scenario
        future_df = utils.make_forecast_df(granularity=granularity_one_digit, category=category, forecast_date=fc_date).drop(
            columns=granular_scenario_variables_ordered_list)
        future_df = future_df.merge(gran_regs_df, how='left', on='ds').dropna()

        # Forecasting the future_df and getting the "weather_bh_corrected distribution" to add onto the monthly model.
        simple_fc = self.granular_model.predict(future_df)
        self.granular_model.granularity = granularity_one_digit
        granular_df = utils.full_forecast_df(
            model=self.granular_model, future=future_df, forecast=simple_fc)
        granular_df['year_month'] = granular_df['ds'].dt.to_period(
            "M").astype(str)
        granular_df['year_month'] = pd.to_datetime(
            granular_df['year_month'], format="%Y-%m")
        granular_df['weather_bh_terms'] = granular_df[['temperature', 'cdd', 'hdd', 'holidays']].sum(
            axis=1) - granular_df[[x for x in granular_df.columns.tolist() if 'lockdown' in x]].sum(axis=1)
        granular_df['corrected_multiplicative_terms'] = 1 + \
            (granular_df['multiplicative_terms'] -
             granular_df['weather_bh_terms'])
        granular_df['normal_multiplicative_terms'] = 1 + \
            granular_df['multiplicative_terms']
        granular_df['yhat_wc_bh'] = granular_df['trend'] * \
            granular_df['corrected_multiplicative_terms']
        granular_df['yhat_real_proportion'] = granular_df.groupby(
            'year_month')['corrected_multiplicative_terms'].transform(lambda x: x.sum())
        granular_df['yhat_real_proportion'] = granular_df['normal_multiplicative_terms'] / \
            granular_df['yhat_real_proportion']
        granular_df.columns = "granular_" + granular_df.columns

        # Getting the columns we are interested in:
        regressors = []
        for reg in granular_scenario_variables_ordered_list:
            regressors.append("granular_"+reg)
            regressors.append("granular_nominal_"+reg)
        granular_df = granular_df.rename(
            columns={"granular_year_month": 'monthly_ds'})
        granular_df = granular_df[[
            'granular_ds', 'monthly_ds', 'granular_yhat_real_proportion', *regressors]]

        return granular_df

    def run_ony_monthly_scenario(self, scenario_number):

        if self.create_scenarios_check == False:
            raise ValueError(
                "Scenarios have not been created, run method 'create_scenarios()")

        granularity_one_digit = 'M'
        category = self.granular_model.history['sector'].unique()[0]
        # Putting the Monthly scenario variables into the "Future DF"

        # Getting the variables' ordered names:
        monthly_model_scenario_variables_ordered_list = []
        monthly_granularity = '1month'
        for var in self.variables.keys():
            if self.variables[var]["granularity"] == monthly_granularity:
                monthly_model_scenario_variables_ordered_list.append(var)

        # Getting the scenario NUMBER 0 regressors and changing the col names to the ordered variable names (from gdp_1 to gdp for example)
        scenarios_df = self.scenarios_dfs
        gran_regs_df = scenarios_df[scenario_number][monthly_granularity].copy(
        )
        fc_date = gran_regs_df['ds'].max()
        scen_cols = [x for x in gran_regs_df.columns if x != 'ds']
        gran_regs_df = gran_regs_df.rename(columns=dict(
            zip(scen_cols, monthly_model_scenario_variables_ordered_list)))

        # Creating the future_df and replacing the "default variables" for the new values from the scenario
        future_df = utils.make_forecast_df(granularity=granularity_one_digit, category=category, forecast_date=fc_date).drop(
            columns=monthly_model_scenario_variables_ordered_list)
        future_df = future_df.merge(gran_regs_df, how='left', on='ds').dropna()

        # Forecasting the future_df and getting the macro scenario.
        simple_fc = self.monthly_model.predict(future_df)
        self.monthly_model.granularity = granularity_one_digit
        monthly_df = utils.full_forecast_df(
            model=self.monthly_model, future=future_df, forecast=simple_fc)
        monthly_df = monthly_df.rename(columns={'ds': 'monthly_ds'})

        # Getting the columns we are interested in:
        regressors = []
        for reg in monthly_model_scenario_variables_ordered_list:
            regressors.append(reg)
            regressors.append("nominal_"+reg)
        monthly_df = monthly_df[['monthly_ds', 'yhat', *regressors]]

        return monthly_df

    def run_one_scenario(self, scenario_number):

        if self.create_scenarios_check == False:
            raise ValueError(
                "Scenarios have not been created, run method 'create_scenarios()")

        scenario_number = scenario_number
        granular_df = self.run_ony_granular_scenario(
            scenario_number=scenario_number)
        monthly_df = self.run_ony_monthly_scenario(
            scenario_number=scenario_number)

        # Final Thingies:
        df = granular_df.merge(monthly_df, how='left', on='monthly_ds')
        df['scenario_yhat'] = df['yhat'] * df['granular_yhat_real_proportion']
        df = df.drop(
            columns=['yhat', 'granular_yhat_real_proportion', 'monthly_ds'])

        return df

    def run_all_scenarios(self):

        if self.create_scenarios_check == False:
            raise ValueError(
                "Scenarios have not been created, run method 'create_scenarios()")

        self.outputs_df = pd.DataFrame()

        for scenario_number in range(len(self.scenarios_dfs)):
            df = self.run_one_scenario(scenario_number=scenario_number)
            df['scenario_number'] = scenario_number + 1
            self.outputs_df = pd.concat([self.outputs_df, df])

        now = datetime.datetime.today().strftime(format="%Y_%m_%d_%H_%M_%S_")
        self.outputs_df = self.outputs_df.reset_index(drop=True)
        self.outputs_df.to_parquet(os.path.join(scenario_outputs, f"{now}scenarios.parquet"))
        
        return self.outputs_df