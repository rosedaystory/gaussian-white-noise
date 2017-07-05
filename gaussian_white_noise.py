
# coding: utf-8

# In[306]:

from sklearn.datasets import make_regression
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn as sk
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
from ipykernel import kernelapp as app


# In[317]:

# make white noise, n_data = number of data, n_length = length of each data

class White(object):
    """
    make gaussian white noise!!
    n_data = number of data
    n_length = length of each data
    """
    
    def __init__(self, n_data, n_length):
        """
        at self.white timeseriese are in direction of row 
        at self.df_white timesiriese are in column direction
        """
        self.n_data = n_data
        self.n_length = n_length
        self.white = np.random.randn(n_data,n_length)
        self.df_white = pd.DataFrame(self.white.T)
        
    def draw(self):
        """
        draw all serieses
        """
        
        plt.plot(range(1,self.n_length+1), self.white.T)
        plt.show()
        
    def drawthe(self, k):
        """
        draw the kth seriese
        """
        plt.plot(range(self.n_length), self.df_white[k-1])
        plt.show()
        
    def std(self,ensemble=True):
        """
        calculate standard deviation
        if ensemble = true, it calculate ensemble std for white noise// default = true
        if ensemble = False, it calculate each timeseriese`s std
        """
        if ensemble ==True:
            k = 0
        elif ensemble == False:
            k = 1
        else:
            print("error!!")
        return self.white.std(axis=k)

    def mean(self,ensemble=True):
        """
        calculate mean
        if ensemble = true, it calculate ensemble mean for white noise// default = true
        if ensemble = False, it calculate each timeseriese`s mean
        """
        if ensemble ==True:
            k = 0
        elif ensemble == False:
            k = 1
        else:
            print("error!!")
        return self.white.mean(axis=k)
    
    def cov(self, k, s):
        """
        calculate cov of kth and sth of timeseriese
        """
        return ((self.white[:,k] - self.white[:,k].mean()) * (self.white[:,s] - self.white[:,s].mean())).mean()
    def cov_all(self):
        """
        calculate all covariance and return it as n by n list
        """
        s = pd.DataFrame(self.white - self.white.mean())
        a = np.random.rand(len(self.white[0]),len(self.white[0]))
        for i in range(len(self.white[0])):
            for j in range(len(self.white[0])):
                   a[i][j]=self.cov(i,j)
        return a
        
        
    
    def lo(self, k, s):
        """
        calculate cov of kth and sth of timeseriese
        """
        return self.cov(k,s) / self.std()[k] / self.std()[s]
    
    def lo_all(self):
        """
        calculate all lo and return it as n by n list
        """
        a = np.random.rand(len(self.white[0]),len(self.white[0]))
        for i in range(len(self.white[0])):
            for j in range(len(self.white[0])):
                   a[i][j]=self.lo(i,j)
        return a


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



