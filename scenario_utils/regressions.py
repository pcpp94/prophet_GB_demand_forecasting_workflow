import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def linear_regressor_model(df, reg):
    regressor_x = df['nominal_'+reg].to_numpy().reshape(-1, 1)
    regressor_y = df[reg].to_numpy().reshape(-1, 1)
    regressor_regr = LinearRegression()
    regressor_model = regressor_regr.fit(regressor_x, regressor_y)
    return regressor_model

# Extrapolating temperature to HDD:


def get_hdd(df, temperature: np.array):
    """
    temperature: numpy array 2D
    """
    X = df.dropna(subset=['y'])[
        'nominal_temperature'].to_numpy().reshape(-1, 1)
    y = df.dropna(subset=['y'])['nominal_hdd'].to_numpy().reshape(-1, 1)
    degree = 2
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X, y)
    prediction = polyreg.predict(temperature.reshape(-1, 1)).flatten()
    prediction[prediction < 0] = 0
    prediction[temperature.flatten() > 18] = 0
    return prediction

# Extrapolating temperature to CDD:


def get_cdd(df, temperature: np.array):
    """
    temperature: numpy array 2D
    """
    X = df.dropna(subset=['y'])[
        'nominal_temperature'].to_numpy().reshape(-1, 1)
    y = df.dropna(subset=['y'])['nominal_cdd'].to_numpy().reshape(-1, 1)
    degree = 3
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(X, y)
    prediction = polyreg.predict(temperature.reshape(-1, 1)).flatten()
    prediction[prediction < 0] = 0
    prediction[temperature.flatten() < 10] = 0
    return prediction
