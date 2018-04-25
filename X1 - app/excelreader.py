#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:56:27 2018

@author: Laurent Rudloff
"""

import numpy as np
import pandas as pd
import os


class PriceReader():
    def __init__(self, filepath):
        self.filepath = filepath
        self.priceset = pd.DataFrame()

    def read_xls(self, filename):
        self.priceset = pd.read_excel(os.path.join(self.filepath, filename))

    def get_prices(self):
        return self.priceset