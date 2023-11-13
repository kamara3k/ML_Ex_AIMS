#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:17:05 2023

@author: kamal
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split


class preprocess:
    def __init__ (self,accept='xs.csv',scale_x='MinMax',scale_y='MinMax',test_size=0.2):
        self.data=pd.read_csv(accept)
        self.scale_x=scale_x
        self.scale_y=scale_y
        self.test_size=test_size
        
    def inpu_data(self):
        data=self.data
        X=data.iloc[:,0:-1] # entire row of the data and the entire column first except last column  
        Y=data.iloc[:,-1]
        
        if self.scale_x=='N': 
            x_scaled=X
        elif self.scale_x=='MinMax':
            x_one_scaler=preprocessing.MinMaxScaler()
            x_one_scaler.fit(X)
            x_scaled=x_one_scaler.transform(X)
        elif self.scale_x=='Standard':
            x_std_scaler=preprocessing.StandardScaler()
            x_std_scaler.fit(X)
            x_scaled=x_std_scaler.transform(X)
        else:
            raise ValueError('***Error: The input X scale should be `N`, `MinMax` or `Standard`')
            
        Y_2d=Y.values.reshape((Y.size, 1)) 
        """
         df = pd.DataFrame (Y_2d)
         filepath = 'my_data.xlsx'
         df.to_excel(filepath, index=False)
         """
        if self.scale_y=='N':
             y_scaled=Y_2d
        elif self.scale_y=='MinMax':
             y_one_scaler=preprocessing.MinMaxScaler()
             y_scaled=y_one_scaler.fit_transform(Y_2d)
        elif self.scale_y=='Standard':
             y_std_scaler=preprocessing.StandardScaler()
             y_scaled=y_std_scaler.fit_transform(Y_2d)
        else:
            raise ValueError('***Error: The input X scale should be `N`, `MinMax` or `Standard`')

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_scaled, y_scaled, test_size=self.test_size, random_state=42)
        return Xtrain, Xtest, Ytrain, Ytest
        