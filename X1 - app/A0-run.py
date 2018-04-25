#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:51:15 2018

@author: Laurent Rudloff
"""
import os
import sys
import time

# Used to create the needed folders
from folderinit import FolderInit
from dxfreader import DxfParser
from reportgen import ReportGen
from datapreparation import DataPrep
from excelreader import PriceReader


# Initiate the folders
folderinit= FolderInit([('..', 'A1 - Estimate', ),
                        ('..', 'B1 - Reports', ),
                        ('..', 'C1 - Train', 'Dxf',),
                        ('..', 'C1 - Train', 'Targets',),
                        ('..', 'X2 - Backup',),
                        ('..', 'Y2 - Sample DXF',),
                        ])

# Early stop if all the folders not pre-existing
folderinit.projectverification()
if not folderinit.valid:
    print()
    sys.exit('All the folders have been initiated, please restart program')


# DXF parser setting and dataset definition
dxfparser = DxfParser(min_edge_length=1,
                        roundigexp=3,
                        max_distance_correction=1.5)

#DataHandler
dataprep = DataPrep(bendrange=[100, 500, 1000, 20000, 5000])

#Excel reader
pricereader = PriceReader('../C1 - Train/Targets')

#Report generator
reportgen = ReportGen('../B1 - Reports', '../X2 - Backup')

#Defines the folder to watch by the daemon (folder, file extension to watch, task to perform)
folderstowatch = [('../A1 - Estimate', 'dxf', 'predict'),
                  ('../C1 - Train/Dxf', 'dxf', 'traindxf'),
                  ('../C1 - Train/Targets', 'xls', 'trainxls')]



scanning = True

while scanning:
    time.sleep(1)
    print('Scanning folders')
    for folder, extension, task in folderstowatch:
        files = [file for file in os.listdir(folder) if file.endswith(extension)]
        if len(files) != 0:


            #Cost prediction request
            if task == 'predict':
                while len(files) != 0:
                    try:
                        details={}
                        try:
                            details = dxfparser.parse(os.path.join(folder, files[0]))
                        except:
                            print('parser Error')
                        finally:
                            os.remove(os.path.join(folder, files[0]))
                        details['filename'] = files[0].split('.')[0]
                        details['bend_count'] = len(details['bend_radius'])
                        reportgen.costreport(details)
                    except:
                        print('something went wrong')
                    files = [file for file in os.listdir(folder) if file.endswith(extension)]


            #Adds a new DXF to the dataset
            elif task == 'traindxf':
                while len(files) != 0:
                    try:
                        details={}
                        try:
                            details = dxfparser.parse(os.path.join(folder, files[0]))
                        except:
                            print('parser Error')
                        finally:
                            os.remove(os.path.join(folder, files[0]))
                    except:
                        print('something went wrong')
                    filename = files[0].split('.')[0]
                    dataprep.add_dxf(filename, details)
                    files = [file for file in os.listdir(folder) if file.endswith(extension)]


            #Triggers the model training from availables DXFs
            elif task == 'trainxls':
                try:
                    pricereader.read_xls(files[0])
                    os.remove(os.path.join(folder, files[0]))
                except:
                    print('the file was probably read only')

                print(dataprep.get_dataset().shape)


