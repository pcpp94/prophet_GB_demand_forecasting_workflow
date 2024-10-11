from git import Repo
import os
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from pandas.plotting import deregister_matplotlib_converters
import seaborn as sns
import datetime
import pandas as pd
from prophet.make_holidays import make_holidays_df
from prophet.plot import *

repo = Repo(os.path.dirname(__file__), search_parent_directories=True)
root_path = repo.git.rev_parse("--show-toplevel")


def reducing(x, y):
    if len(str(x)) == 19:
        return str(x)[0:10] + ',' + str(y)[0:10]
    else:
        return str(x) + ',' + str(y)[0:10]


__all__ = ["root_path"]
