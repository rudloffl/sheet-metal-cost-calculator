#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 21:56:27 2018

@author: Laurent Rudloff
"""

import datetime
from string import Template
##https://docs.python.org/3/library/string.html#template-strings
import os

class ReportGen():
    def __init__(self, reportsloc, sampleloc):
        self.reportsloc = reportsloc

        with open(os.path.join(sampleloc, 'part-report.txt')) as file:
            self.reportsample = Template(file.read())

        with open(os.path.join(sampleloc, 'train-report.txt')) as file:
            self.trainsample = Template(file.read())

        with open(os.path.join(sampleloc, 'error-report.txt')) as file:
            self.errorsample = Template(file.read())

    def costreport(self, details):
        filename = details.get('filename', 'error')
        now = datetime.datetime.now()
        details['date'] = now.strftime("%Y-%m-%d %H:%M")

        report = self.reportsample.safe_substitute(details)
        with open(os.path.join(self.reportsloc, f'{filename}.txt'), 'w') as file:
        	file.write(report)

    def fitreport(self, **details):
        toreport=''
        #print(details.keys())
        for target in details.keys():
            report = self.trainsample.safe_substitute({**details[target]['details'], 'target':target})
            toreport = toreport + '===============================================\n' + report 

            #for (th, coart), val in details[target]['worst'].iteritems():
            #    toreport = toreport + '{:.02d} --- {} --- {:.003d}\n'.format(th, coart, val)

        filename = 'train_report'
        now = datetime.datetime.now()
        stamp = now.strftime("%Y-%m-%d_%H-%M")
        with open(os.path.join(self.reportsloc, f'{filename}_{stamp}.txt'), 'w') as file:
            file.write(toreport)


    def errorreport(self, details):
        filename = details.get('filename', 'error')
        now = datetime.datetime.now()
        details['date'] = now.strftime("%Y-%m-%d %H:%M")

        report = self.errorsample.safe_substitute(details)
        with open(os.path.join(self.reportsloc, f'{filename}.txt'), 'w') as file:
            file.write(report)




if __name__ == "__main__":
    reportgen = ReportGen('../B1 - Reports', '../X2 - Backup')
    reportgen.errorreport({'filename':'test'})
