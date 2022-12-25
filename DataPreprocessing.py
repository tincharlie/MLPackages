import os
import numpy as np
import pandas as pd
import warnings

import plotly
import plotly.graph_objs as go
import plotly.io as pio
import plotly.express as px
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf

#machine learning libraries:
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split, KFold, cross_val_score
from sklearn.preprocessing  import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb



class Information:
    """
    This is for information of data.
    """
    def __init__(self):
        print()
        print("Info object is created")
        print()

    def get_missing_value(self, data):
        """
        This  function finds the missing values in the dataset.
        ...
        :param data: Pandas DataFrame
        The data you want to see information about


        :return: A pandas series contains the missing values in descending or
        """
        missing_values = data.isnull().sum()

        missing_values = missing_values.sort_values(ascending=False)

        return missing_values

    def _info_(self, data):
        self.data = data
        feature_dtypes = self.data.dtypes
        self.missing_value( )
