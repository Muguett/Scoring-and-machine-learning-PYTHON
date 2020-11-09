#!/usr/bin/env python
# coding: utf-8

# In[1]:


import statsmodels.api as sm
from sklearn import datasets 
data = datasets.load_boston() 


# In[2]:


print(data.DESCR)


# In[4]:


import numpy as np
import pandas as pd

#define the data/predictors as the pre-set feature nammes
df = pd.DataFrame(data.data, columns=data.feature_names)

#put the target (housing value--- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=['MEDV'])


# In[7]:


df


# In[14]:


### without a constant
import statsmodels.api as sm

X = df['RM']
Y = target['MEDV']

model = sm.OLS(Y,X)
results = model.fit()
predictions = results.predict(X)#make the predictions by the model

#print out statistics
results.summary()


# In[15]:


import statsmodels.api as sm

X = df['RM']
Y = target['MEDV']
X = sm.add_constant(X)

model = sm.OLS(Y,X).fit()

model.summary()


# In[ ]:


'''' 
price increase with the size
the rate of increase is change
If we had more smaller houses than bigger one, this aspect hasn't be computured 
we check the data set 
then make a non linear model
''''


# In[17]:


X = df[['RM','LSTAT']]
Y = target['MEDV']
#X = sm.add_constant(X) <- we do not add 

model = sm.OLS(Y,X).fit()
predictions = model.predict(X)

model.summary()


# In[19]:


"""
Main issue of R^2 -> the r^2 increases even (if we add non significant) increase the number of variable 
What we check when we do OLS? ??
correlation =0 and still have depandent variables -> incase of non linerity yes 
"""


# In[24]:


from sklearn import datasets ### imports datasets from scikit-learn
data=datasets.load_boston() ### loads boston daraset from datasets library


# In[25]:


X = df
Y = target['MEDV']


# In[27]:


import sklearn
from sklearn import linear_model

lm = linear_model.LinearRegression()
model = lm.fit(X,Y)


# In[28]:


predictions = lm.predict(X)
print(predictions[0:5])


# In[29]:


lm.score(X,Y)


# In[30]:


lm.coef_


# In[31]:


get_ipython().system('pip install linearmodels')


# In[40]:


####Principal component analysis
get_ipython().system('pip install sklearn.processing ')
get_ipython().system('pip install --upgrade pip')


# In[60]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[83]:


data = pd.read_csv('/Users/mugefirsat/Downloads/peach_spectra_brix.csv')
X = data.values[:,1:]
y = data['Brix']
wl = np.arange(1100,2300,2)
#Plot abbsorbance spectra
with plt.style.context(('ggplot')):
     plt.plot(wl, X,y)
     plt.xlabel('Waveleght')
     plt.ylabel('Absorbence')
plt.show()


# In[62]:


#select how much of variance you will use


# In[65]:


def pcr(X,Y,pc):
    '''Principal component regression in python '''
    '''Step 1: PCA on input data '''
    #define the PCA object
    pca= PCA()
    #Preprocessing(1): first derivator
    d1X = savgol_filter(X,25,polyorder =5, deriv=1)
    #Preprocess(2) Standardize features by removingthe mean and scaling to unit variance
    Xstd=StandardScaler().fit_transform(d1X[:,:])
    #Run PCA producing the reduced variable Xred and selecte the first pc components 
    Xreg= pca.fit_transform(Xstd)[:,:pc]
    ''' Step 2: regression on selected principal commponents'''
    #Create Linear regression obect
    regr = linear_model.LinearRegression()
    #Fit
    regr.fit(Xreg,Y)
    #Calibration
    y_c = regr.predict(Xreg)
    #Cross validation
    y_cv = cross_val_predict(regr, Xreg, Y, cv=10)
    #Calculate scores for calibration and cross-validation
    score_c = r2_score(Y,y_c)
    score_cv = r2_score(Y,y_c)
    #Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(Y, y_c)
    mse_cv = mean_squared_error(Y,y_c)
    return(y_cv, score_c, score_cv, mse_c,mse_cv)


# In[84]:


predicted, r2r, r2cv,mser, mscv = pcr(X,y, pc=10)


# In[85]:


print(r2cv)


# In[ ]:




