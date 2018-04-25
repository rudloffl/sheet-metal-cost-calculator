#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:51:15 2018

@author: Laurent Rudloff
"""
import numpy as np
import pandas as pd

class DataPrep(object):
    """Dataset formatting"""
    def __init__(self, bendrange=[100, 500, 1000, 20000, 5000]):
        self.bendrange = bendrange
        self.columns = ['cut_length', 'num_closed_patterns', 'num_open_patterns', 'total_area', 'minimum_rectangle_area',
                         'minimum_rectangle_dim1', 'minimum_rectangle_dim2', 'convex_hull_area', 'no_hole_area',
                         'thickness', 'unit', 'material', 'bend_radius', 'bend_angle', 'bend_direction',
                         'deformation_length', 'possible_imperfection', 'bend_bend_distance', 'bend_bend_angle',
                         'merged_bend', 'punch_length', 'radius_approx', 'bend_edge_distance', 'bend_edge_angle',
                         'bend_edge_length']

        self.dataset = pd.DataFrame(columns=self.columns)
        self.initdataset = True

    def transform(self, dataset):
        pass

    def fit(self, dataset):
        pass

    def add_dxf(self, filename, details):
        if self.initdataset:
            self.dataset = pd.DataFrame(columns=self.columns)
            self.initdataset = False
        self.dataset.loc[filename] = [details.get(x, np.nan) for x in self.dataset.columns]
        print(f'{filename} added to the dataset')

    def get_dataset(self):
        self.initdataset = True
        return self.dataset
