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

    def costreport(self, details):
        filename = details.get('filename', 'error')
        now = datetime.datetime.now()
        details['date'] = now.strftime("%Y-%m-%d %H:%M")

        report = self.reportsample.safe_substitute(details)
        with open(os.path.join(self.reportsloc, f'{filename}.txt'), 'w') as file:
        	file.write(report)

    def fitreport(self, details):
        pass

    def errorreport(self, details):
     	pass




if __name__ == "__main__":
    pass
