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
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.externals import joblib
import os
import pickle
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from sklearn.model_selection import cross_val_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

class Costcalculator(object):
    """Cost prediction system"""
    def __init__(self, target, qhigh=.95, qlow=.02, backupfolder='../X2 - Backup'):
        self.backupfolder = backupfolder
        self.target = target
        try:
            self.load_model()
        except: #New model
            self.limitedset = []
            self.error = pd.DataFrame()
            self.worst = pd.DataFrame()
            self.qlow = qlow
            self.qhigh = qhigh
            self.estimator = lgb.LGBMRegressor(objective='regression',
                        n_estimators=300,
                        random_state=0,
                        verbose=5,
                       )
            self.lmbda = 0
            self.details = {}
            self.scaler = StandardScaler()

    def fit(self, dataset, max_evals=100):
        """Method used to train model"""

        #1 - Columns used definition depending on target
        self.limitedset = ['cut_length', 'num_closed_patterns', 'num_open_patterns', 'total_area', 'minimum_rectangle_area',
              'minimum_rectangle_dim1', 'minimum_rectangle_dim2', 'convex_hull_area', 'no_hole_area',]
        ## 1 Bend as a float
        self.limitedset.append('thickness')
        ## 2 Different counts
        self.limitedset.extend(['radius_count', 'direction_count', 'bend_count'])
        if self.target == 'VALAJOUT':
            self.limitedset.extend([x for x in dataset.columns.values if x.startswith('bendlength_')])


        #2 - Outlier suppression
        if self.target =='VAMAT':
            dataset['DISCRIMINANT'] = dataset['VAMAT'] / dataset['minimum_rectangle_area']
            df=dataset[['thickness', 'DISCRIMINANT']].groupby('thickness').quantile([self.qlow,self.qhigh])

            for (thickness, limit), value in df.iterrows():
                tresh = value.values[0]
                if limit == self.qlow:
                    mask = np.logical_and(dataset['thickness'] == thickness, dataset[df.columns.values[0]] < tresh)
                    dataset = dataset[np.logical_not(mask)]
                elif limit == self.qhigh:
                    mask = np.logical_and(dataset['thickness'] == thickness, dataset[df.columns.values[0]] > tresh)
                    dataset = dataset[np.logical_not(mask)]


        elif self.target == 'VALAJOUT':
            dataset['DISCRIMINANT'] = dataset['VALAJOUT'] / (dataset['bend_count']+1)
            df=dataset[['thickness', 'DISCRIMINANT', 'bend_count']].groupby(['thickness', 'bend_count']).quantile([self.qlow,self.qhigh])

            for (thickness, bend_count, limit), value in df.iterrows():
                tresh = value.values[0]
                mask1 = np.logical_and(dataset['thickness'] == thickness, dataset['bend_count'] == bend_count)
                if limit == self.qlow:
                    mask = np.logical_and(mask1, dataset[df.columns.values[0]] < tresh)
                    dataset = dataset[np.logical_not(mask)]
                elif limit == self.qhigh:
                    mask = np.logical_and(mask1, dataset[df.columns.values[0]] > tresh)
                    dataset = dataset[np.logical_not(mask)]
        dataset = dataset.fillna(0)
        self.details['datasetsize']=dataset.shape[0]

        #3 - Train - Test creation

        X_train, X_test, y_train, y_test = train_test_split(dataset, dataset[[self.target]], test_size=0.3, random_state=0 ,stratify=dataset['bend_group'])
        self.details['trainsize']=X_train.shape[0]
        self.details['testsize'] =X_test.shape[0]

        #4 - BOX-COX
        _, self.lmbda = stats.boxcox(y_train[self.target], lmbda=None)
        y_train['logtarget'] = y_train[self.target].apply(lambda x: stats.boxcox(x, self.lmbda))
        y_test['logtarget'] = y_test[self.target].apply(lambda x: stats.boxcox(x, self.lmbda))

        #5 - scaling
        #print(dataset[self.limitedset].isnull().sum())
        X_train_scaled = self.scaler.fit_transform(X_train[self.limitedset])
        X_test_scaled  = self.scaler.transform(X_test[self.limitedset])

        #6 - Model training
        def objective(params):
            params = {
                'num_leaves': int(params['num_leaves']),
                'min_data_in_leaf': int(params['min_data_in_leaf']),
                'min_child_weight': params['min_child_weight'],
                #'n_estimators': int(params['n_estimators']),
                'colsample_bytree': params['colsample_bytree'],
                'bagging_fraction': params['bagging_fraction'],
                'bagging_freq': params['bagging_freq'],
                'reg_alpha': params['reg_alpha'],
                'reg_lambda': params['reg_lambda'],
                'max_depth':int(params['max_depth']),
                'learning_rate':params['learning_rate'],
                }
            
            clf = lgb.LGBMRegressor(objective='regression', n_estimators=200, **params)

            
            score = cross_val_score(clf, X_train_scaled, y_train['logtarget'], scoring='neg_mean_squared_error', cv=3, n_jobs=-2).mean()
            print("MSE {:.3f} - params {}".format(score, params))
            return -score

        space = {
            'num_leaves': hp.uniform('num_leaves', 5, 40),
            'min_data_in_leaf': hp.uniform('min_data_in_leaf',10, 40),
            'min_child_weight': hp.uniform('min_child_weight', 0.001, 20),
            #'n_estimators': hp.uniform('n_estimators', 100, 500),
            'colsample_bytree': hp.uniform('colsample_bytree', 0., 1.0),
            'bagging_fraction': hp.uniform('bagging_fraction', 0., 1.0),
            'bagging_freq': hp.randint('bagging_freq', 15),
            'reg_alpha': hp.loguniform('reg_alpha', -3, 3),
            'reg_lambda': hp.loguniform('reg_lambda', -3, 3),
            'max_depth': hp.uniform('max_depth', 3, 15),
            'learning_rate': hp.uniform('learning_rate', 0.001, .1),
        }

        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_evals)
        params = {
        'num_leaves': int(best['num_leaves']),
        'min_data_in_leaf': int(best['min_data_in_leaf']),
        'min_child_weight': best['min_child_weight'],
        #'n_estimators': int(best['n_estimators']),
        'colsample_bytree': best['colsample_bytree'],
        'bagging_fraction': best['bagging_fraction'],
        'bagging_freq': best['bagging_freq'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda'],
        'max_depth':int(best['max_depth']),
        'learning_rate':best['learning_rate'],
        }

        self.estimator.set_params(**params)

        self.estimator.fit(X_train_scaled, y_train['logtarget'])









        #7 - Performance calculation
        prediction_lgbm_log = inv_boxcox(self.estimator.predict(X_test_scaled), self.lmbda)

        resultset = pd.DataFrame({'target': y_test[self.target],
                          'pred_lgbm':prediction_lgbm_log,
                            }, index=y_test.index)
        self.details['MAPE'] = mean_absolute_percentage_error(resultset['target'], resultset['pred_lgbm'])
        self.details['MSE']  = mean_squared_error(resultset['target'], resultset['pred_lgbm'])

        resultset['MSE_lgbm'] = (resultset['target']-resultset['pred_lgbm'])**2
        resultset['MAPE_lgbm'] = resultset.apply(lambda x: mean_absolute_percentage_error(x['target'], x['pred_lgbm']), axis=1)

        comparisonset = pd.merge(resultset.reset_index(), X_test.reset_index(), on='COART', how='inner')

        if self.target == 'VALAJOUT':
            df = comparisonset[['MSE_lgbm','MAPE_lgbm','thickness', 'bend_group', 'COART']].groupby(['thickness', 'bend_group'])

        elif self.target == 'VAMAT':
            df = comparisonset[['MSE_lgbm', 'MAPE_lgbm', 'thickness', 'COART']].groupby(['thickness'])

        ## Error calculation
        self.error = df.agg({'MSE_lgbm': ['mean', 'max'],
                             'MAPE_lgbm': ['mean', 'max'],
                             'COART':['count'] })

        ## Worst parts calculation
        df = comparisonset[['MSE_lgbm', 'COART', 'thickness']]
        df.set_index('COART', inplace=True)
        self.worst = df.groupby('thickness')['MSE_lgbm'].nlargest(5)

        self.save_model()

        return self

    def predict(self, dataset, y=None):
        X_scaled  = self.scaler.transform(dataset[self.limitedset])
        ylog = self.estimator.predict(X_scaled)
        return inv_boxcox(ylog, self.lmbda)[0]

    def get_details(self):
        return self.details

    def get_performance(self):
        return self.error

    def get_worst(self): 
        return self.worst

    def save_model(self):
        self.error.to_csv(os.path.join(self.backupfolder, 'error-{}.csv'.format(self.target)))
        self.worst.to_csv(os.path.join(self.backupfolder, 'worst-{}.csv'.format(self.target)))
        joblib.dump(self.estimator, os.path.join(self.backupfolder, 'estimator-{}.pkl'.format(self.target)))
        joblib.dump(self.scaler, os.path.join(self.backupfolder, 'scaler-{}.pkl'.format(self.target)))
        tosave = {'limitedset':self.limitedset,
                    'qlow':self.qlow,
                    'qhigh':self.qhigh,
                    'lmbda':self.lmbda,
                    'details':self.details}
        pickle.dump( tosave, open( os.path.join(self.backupfolder, 'details-{}.pkl'.format(self.target)), 'wb' ) )
        return self

    def load_model(self):
        if self.target == 'VAMAT':
            self.error = pd.read_csv(os.path.join(self.backupfolder, 'error-{}.csv'.format(self.target)), index_col=0, header=[0,1])
            self.worst = pd.read_csv(os.path.join(self.backupfolder, 'worst-{}.csv'.format(self.target)), index_col=[0,1], squeeze=True, header=None)
        else:
            self.error = pd.read_csv(os.path.join(self.backupfolder, 'error-{}.csv'.format(self.target)), index_col=[0,1], header=[0,1])
            self.worst = pd.read_csv(os.path.join(self.backupfolder, 'worst-{}.csv'.format(self.target)), index_col=[0,1], squeeze=True, header=None)
        self.estimator = joblib.load(os.path.join(self.backupfolder, 'estimator-{}.pkl'.format(self.target)))
        self.scaler = joblib.load(os.path.join(self.backupfolder, 'scaler-{}.pkl'.format(self.target)))
        saved = pickle.load( open( os.path.join(self.backupfolder, 'details-{}.pkl'.format(self.target)), 'rb' ) )
        self.limitedset = saved['limitedset']
        self.qlow = saved['qlow']
        self.qhigh = saved['qhigh']
        self.lmbda = saved['lmbda']
        self.details = saved['details']
        return self


if __name__ == '__main__':
    calcvamat    = Costcalculator(qhigh=.95, qlow=.02, target='VAMAT')
    #dataset = pd.read_csv('../Y3 - Sample data/Z2 - datasetprepclean.csv', index_col = 0)
    dataset = pd.read_csv('../X2 - Backup/trainset.csv', index_col = 0)
    print(calcvamat.details)
    calcvamat.fit(dataset)
    #calcvamat.save_model()
    
    #print(calcvamat.details)
    calcvalajout = Costcalculator(qhigh=.95, qlow=.02, target='VALAJOUT')
    calcvalajout.fit(dataset)
    print(calcvalajout.details)


