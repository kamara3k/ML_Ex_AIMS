#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:08:42 2023

@author: kamal
"""

from postprocess import PostProcess
from MLregressor import MLregressor
from preprocess import preprocess
import pandas as pd

prep = preprocess(accept='xs.csv',test_size=0.15)
Xtrain, Xtest, Ytrain, Ytest=prep.inpu_data()

regr = MLregressor(Xtrain,Ytrain)
lr_model = regr.run_lr()
lr_pred=lr_model().predict(Xtest)

postp = PostProcess(lr_pred,Ytest)
print('Linear Regression Metrics:')
postp.print_metrics()

# *************************************

rfr_model = regr.run_rfr()
rfr_pred=rfr_model().predict(Xtest)

postp = PostProcess(rfr_pred,Ytest)
print('Random Forest Regression Metrics:')
postp.print_metrics()

# *************************************

nn_model,nn_history = regr.run_nn()
nn_pred=nn_model.predict(Xtest)

postp = PostProcess(nn_pred,Ytest)
print('Neural Network Metrics:')
postp.print_metrics()