#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:56:46 2018

@author: cricket
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


class Costcalculator(object):
    """Cost prediction system"""
    def __init__(self, backup, colnames):
        self.backup = backup
        self.colnames = colnames
        self.performance = None
        self.error = pd.DataFrame()

    def fit(self, X, y):
        pass

    def predict(self, X, y=None):
        pass

    def get_test_accuracy(self):
        pass

    def get_worst(self):
        pass