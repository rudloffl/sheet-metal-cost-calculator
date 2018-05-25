#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:51:15 2018

@author: Laurent Rudloff
"""
import numpy as np
import pandas as pd
from ast import literal_eval

class DataPrep(object):
    """Dataset formatting"""
    def __init__(self, bendcount=5):
        self.columns = ['cut_length', 'num_closed_patterns', 'num_open_patterns', 'total_area', 'minimum_rectangle_area',
                         'minimum_rectangle_dim1', 'minimum_rectangle_dim2', 'convex_hull_area', 'no_hole_area',
                         'thickness', 'unit', 'material', 'bend_radius', 'bend_angle', 'bend_direction',
                         'deformation_length', 'possible_imperfection', 'bend_bend_distance', 'bend_bend_angle',
                         'merged_bend', 'punch_length', 'radius_approx', 'bend_edge_distance', 'bend_edge_angle',
                         'bend_edge_length']

        self.dataset = pd.DataFrame(columns=self.columns)
        self.bendcount = bendcount
        self.todrop = ['index', 'level_0', 'unit', 'material', 'bend_radius', 'bend_angle', 'bend_direction', 'deformation_length', 'bend_bend_distance',
          'bend_bend_angle', 'merged_bend', 'punch_length', 'bend_edge_distance', 'bend_edge_angle',
          'bend_edge_length', 'project', 'VERSION', 'COSTE', 'TPPRX', 'DTPRP', 'VALIDDATE', 'possible_imperfection'
         ]

    def format_dataset(self, dataset):
        toconvert = ['bend_radius', 'bend_angle', 'bend_direction', 'deformation_length',
            'bend_bend_distance', 'bend_bend_angle', 'merged_bend', 'punch_length', 'bend_edge_distance',
            'bend_edge_angle', 'bend_edge_length']
        toconvert_l2 = ['bend_bend_distance', 'bend_bend_angle', 'bend_edge_distance', 'bend_edge_angle', 'bend_edge_length']
        
        #Drop all the errors
        #print(dataset.shape)
        #print(dataset.columns.values)

        #Dataformatting
        tointeger = ['num_closed_patterns', 'num_open_patterns']
        tofloat = ['cut_length', 'total_area', 'minimum_rectangle_area', 'minimum_rectangle_dim1',
           'minimum_rectangle_dim2', 'convex_hull_area', 'no_hole_area', 'thickness',]

        #for column in tointeger:
        #    dataset[column] = pd.to_numeric(dataset[column], downcast='integer', errors='ignore')
        #for column in tofloat:
        #   dataset[column] = pd.to_numeric(dataset[column], downcast='float', errors='ignore')

        dataset['temp'] = dataset['cut_length'].astype(object)
        dataset = dataset[dataset['temp'] != 'ERROR']

        #Small parts exclusion
        dataset['total_area'] = dataset['total_area'].astype('float64')
        dataset = dataset[dataset['total_area'] > 20*20]

        #COART creation
        dataset = dataset.reset_index()
        dataset['COART'] = dataset['index'].apply(lambda x: x.split('_')[0].upper()[:8])
        
        mask = dataset['COART'].duplicated(keep='last')

        #print('Number of dupplicates : ', dataset[mask].shape[0])
        dataset.set_index('COART', inplace=True)

        #Drop all the files with no thickness
        dataset = dataset.dropna(subset=['thickness'])

        # UP-DOWN conversion
        dataset['bend_direction'] = dataset['bend_direction'].apply(lambda cell: [1 if x=='UP' else -1 for x in cell])



        #Order of bends calculation
        ordermask = dataset['deformation_length'].apply(np.argsort).tolist()
        #print(ordermask)

        #Feature re-ordering in dataset
        for column in toconvert:
            #print(column, 'has been reordered')
            dataset[column] = [np.array(arr)[order[::-1]] if len(arr)!=0 else np.array([]) for arr, order in zip(dataset[column], ordermask)]

        #Bends unpacking G2

        #Unpack first level
        for index in range(self.bendcount):
            for bend_info in toconvert:
                dataset['bendlength_{}_{:02d}'.format(bend_info, index)] = dataset[bend_info].apply(lambda x: x[index] if len(x)>=index+1 else [])

        #Create the list of second level
        tounpack = []
        for index in range(self.bendcount):
            tounpack.extend(['bendlength_{}_{:02d}'.format(x, index) for x in toconvert_l2])    
            
        for column in tounpack:
            index = int(column[-2:])
            rootname = column[:-3]
            dataset['{}_mean_{:02d}'.format(rootname, index)] = dataset[column].apply(lambda x: np.nanmean(x) if len(x) !=0 else [])
            dataset['{}_std_{:02d}'.format(rootname, index)]  = dataset[column].apply(lambda x: np.nanstd(x) if len(x) !=0 else [])
            dataset['{}_min_{:02d}'.format(rootname, index)]  = dataset[column].apply(lambda x: np.nanmin(x) if len(x) !=0 else [])
            dataset['{}_max_{:02d}'.format(rootname, index)]  = dataset[column].apply(lambda x: np.nanmax(x) if len(x) !=0 else [])

        dataset = dataset.drop(tounpack, axis=1)

        #Bend group column and bend count
        def bendgroupcalc(entry):
            if entry == 0:
                return '0'
            elif entry <= 2:
                return '1-2'
            elif entry <= 5:
                return '3-5'
            else:
                return '6+'
            return 'None'

        dataset['bend_count'] = dataset['bend_radius'].apply(lambda x: len(x))
        dataset['bend_group'] = dataset['bend_count'].apply(bendgroupcalc)

        #Some other counts
        dataset['radius_count'] = dataset['bend_radius'].apply(lambda x: len(set(x)))
        dataset['direction_count'] = dataset['bend_direction'].apply(lambda x: len(set(x)))

        #replacing all the empty brackets
        for column in dataset.columns.values:
            dataset[column] = dataset[column].apply(lambda x: 0 if type(x)==list else x)

        return dataset

    def format_fit(self, priceset, minocc=2):
        dataset = self.format_dataset(self.dataset)

        todrop = []
        df = dataset['thickness'].reset_index().groupby('thickness').count()
        for index, occ in df.iterrows():
            if occ.COART <= minocc:
                todrop.append(index)

        for thickness in todrop:
            dataset = dataset[dataset['thickness'] != thickness]
        print(f'thickness with less than {minocc} occurences : ', todrop)

        dataset = dataset.reset_index()
        priceset = priceset.reset_index()

        #Merge
        Kset = pd.merge(dataset, priceset, on='COART', how='inner')

        #Only S355 kept
        mask = [x.startswith('S355MC') for x in Kset['material'].tolist()]
        Kset = Kset[mask]

        #np.nan cleaning
        subset = [x for x in Kset.columns.values if x.startswith('bendlength_')]
        Kset = Kset.dropna(subset=subset)


        tokeep = [x for x in Kset.columns.values if x not in self.todrop]

        return Kset[tokeep]



    def add_dxf(self, filename, details):
        self.dataset.loc[filename] = [details.get(x, 'ERROR') for x in self.dataset.columns]
        print(f'{filename} added to the dataset')
        self.dataset.to_csv('../X2 - Backup/extract.csv')

    def get_part_specs(self, filename, details):
        partspec = pd.DataFrame(columns=self.columns)
        partspec.loc[filename] = [details.get(x, 0) for x in self.dataset.columns]
        return self.format_dataset(partspec)

    def reset_dataset(self):
        self.dataset = pd.DataFrame(columns=self.columns)

if __name__ == '__main__':
    dataprep = DataPrep()
    #dataprep.dataset = pd.read_csv('../Y3 - Sample data/Z1 - dataset.csv', index_col = 0,)
    print(dataprep.format_dataset(pd.read_csv('extract.csv', index_col = 0,)))
