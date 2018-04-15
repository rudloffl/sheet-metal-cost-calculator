#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:56:46 2018

@author: cricket
"""

import numpy as np
import pandas as pd



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