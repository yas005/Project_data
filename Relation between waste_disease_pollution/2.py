# -*- coding: utf-8 -*-
"""
Created on Sat May 29 17:14:57 2021

@author: user
"""

import pandas as pd
df = pd.read_csv("D:/data_2010_2019_fin.csv")
df.head()

from sklearn.model_selection import train_test_split
x = df[["Unnamed: 0", "year","combustible waste", "ozone", "sulfur dioxide",
               "nitrogen dioxide", "carbon monoxide"]]
y = df[["parkinson's disease","asthma", "chronic bronchitis","acute myocardial infarction","angina pectoris"]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train, y_train) 

