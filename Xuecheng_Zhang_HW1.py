# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
import statistics 
#part a
np.random.RandomState(1) # Set the random seed
x_1 = np.random.uniform(0, 1, size = (100, )) # X  from the uniform distribution
e_1 = np.random.normal(0,1,size=(100,)) # the residual from the given Gaussin distribution
y_1 = 5 + 3*x_1 + e_1 #  the model from given relation
data1= pd.DataFrame({'x':x_1,'y':y_1})
sns.regplot(data1.x, 
            data1.y, order=1, ci=None,
            scatter_kws={'color':'r', 's':9}) #https://seaborn.pydata.org/generated/seaborn.regplot.html


#part b
places=[]#define an emoty list
for i in range(1,1000):
    x_2 = np.random.uniform(0, 1, size = (100, )) # X  from the uniform distribution
    e_2 = np.random.normal(0,1,size=(100,)) # the residual from the given Gaussin distribution
    y_2 = 5 + 3*x_2 + e_2
    data2= pd.DataFrame({'x':x_2,'y':y_2})
    regr = skl_lm.LinearRegression()
    X = scale(data2.x, with_mean=True, with_std=False).reshape(-1,1)
    y = data2.y
    regr.fit(X,y)
    Beta= regr.coef_[0]
    places.append(Beta)
data_beta=pd.DataFrame({'beta':places})
 
betamean=statistics.mean(data_beta.beta)#find the mean of beta
print('The mean of beta is:',betamean)
#
plt.hist(data_beta.beta, bins = 100)# I set bins is 100


#part c)
places_2=[]#define an emoty list
for i in range(1,1000):
    x_3 = np.random.uniform(0, 1, size = (100, )) # X  from the uniform distribution
    e_3 = np.random.standard_cauchy(100) # the residual from the given Gaussin distribution
    y_3 = 5 + 3*x_3 + e_3
    data3= pd.DataFrame({'x':x_3,'y':y_3})
    regr = skl_lm.LinearRegression()
    X = scale(data3.x, with_mean=True, with_std=False).reshape(-1,1)
    y = data3.y
    regr.fit(X,y)
    Beta= regr.coef_[0]
    places_2.append(Beta)
data_beta3=pd.DataFrame({'beta':places_2})

betamean2=statistics.mean(data_beta3.beta)#find the mean of beta
print('The mean of beta is:',betamean2)
#
plt.hist(data_beta3.beta, bins = 100)# I set bins is 100


