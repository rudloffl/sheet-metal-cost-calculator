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
from costcalculator import Costcalculator
from scipy.stats import skew


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
dataprep = DataPrep(bendcount=5)

#Excel reader
pricereader = PriceReader('../C1 - Train/Targets')

#Report generator
reportgen = ReportGen('../B1 - Reports', '../X2 - Backup')

#Defines the folder to watch by the daemon (folder, file extension to watch, task to perform)
folderstowatch = [('../A1 - Estimate', 'dxf', 'predict'),
                  ('../C1 - Train/Dxf', 'dxf', 'traindxf'),
                  ('../C1 - Train/Targets', 'xls', 'trainxls')]

#Instantiate the cost calculator
calcvamat    = Costcalculator(qhigh=.95, qlow=.02, target='VAMAT')
calcvalajout = Costcalculator(qhigh=.95, qlow=.02, target='VALAJOUT')
if len(calcvamat.details) != 0 and len(calcvalajout.details) != 0:
    reportgen.fitreport(VAMAT={'details':calcvamat.details,
                               'worst':calcvamat.worst,},
                        VALAJOUT={'details':calcvalajout.details,
                               'worst':calcvalajout.worst,},)

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
                            details['filename'] = files[0].split('.')[0]
                            partinfo = dataprep.get_part_specs(details['filename'], details)
                            details['VAMAT'] = calcvamat.predict(partinfo)
                            thickness = details['thickness']
                            error = calcvamat.error
                            try:
                                details['VAMAT_MSE'] = error.loc[thickness]['MSE_lgbm', 'mean']
                                details['VAMAT_MAPE'] = error.loc[thickness]['MAPE_lgbm', 'mean']
                            except KeyError:
                                details['VAMAT_MSE'] = 'Unknown Thickness'
                                details['VAMAT_MAPE'] = 'Unknown Thickness'

                            details['VALAJOUT'] = calcvalajout.predict(partinfo)
                            error = calcvalajout.error
                            bendgr = partinfo.iloc[0]['bend_group']
                            try:
                                details['VALAJOUT_MSE'] = error.loc[thickness, bendgr]['MSE_lgbm', 'mean']
                                details['VALAJOUT_MAPE'] = error.loc[thickness, bendgr]['MAPE_lgbm', 'mean']
                            except KeyError:
                                details['VALAJOUT_MSE'] = 'Unknown Thickness'
                                details['VALAJOUT_MAPE'] = 'Unknown Thickness'
                            
                            details['VALTOT'] = details['VAMAT']+details['VALAJOUT']
                            details['bend_count'] = partinfo.iloc[0]['bend_count']
                            reportgen.costreport(details)
                        except:
                            print('parser Error')
                            details['filename'] = files[0].split('.')[0]
                            reportgen.errorreport(details)
                        finally:
                            os.remove(os.path.join(folder, files[0]))
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
                #try:
                priceset = pricereader.read_xls(files[0])
                os.remove(os.path.join(folder, files[0]))
                asmset = dataprep.format_fit(priceset)
                #except:
                #    print('the file was probably read only')
                asmset.to_csv('../X2 - Backup/trainset.csv')
                dataprep.reset_dataset()

                


