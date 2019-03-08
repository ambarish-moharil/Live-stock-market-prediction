
# coding: utf-8

# In[ ]:


import pandas as pd
import quandl as q
import mat


# In[ ]:


df = q.get('WIKI/GOOGL')


# In[ ]:


df.head()


# In[ ]:


df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]


# In[ ]:


df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'] *100


# In[ ]:


df['PCT_CHNG'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] *100


# In[ ]:


df.head()


# In[ ]:


forecast_col = 'Adj. Close'
df.fillna(value= -99999, inplace= True)
forcast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forcast_out)


# In[ ]:


df.head()


# In[ ]:


df.dropna(inplace= True)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing


# In[ ]:


X = np.array(df.drop(['label'],1))
y = np. array(df['label'])
X = preprocessing.scale(X)


# In[ ]:


clf = LinearRegression()


# In[ ]:


X_train, X_test, y_test, y_train = train_test_split(X,y, test_size=0.2)


# In[ ]:


print(len(X), len(y))


# In[ ]:


print(len(X_train), len(y_train))


# In[ ]:


clf. fit(X_train, y_train)

