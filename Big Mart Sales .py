#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns


# In[5]:


df_train= pd.read_csv(r'C:\Users\HP-PC\Documents\Big Mart Sales\Train.csv')
df_test= pd.read_csv(r'C:\Users\HP-PC\Documents\Big Mart Sales\Test.csv')


# In[6]:


df_train


# In[7]:


df_train.isnull()


# In[8]:


df_train.isnull().sum()


# In[9]:


df_train.shape


# In[10]:


df_test.isnull().sum()


# In[11]:


df_train.info()


# In[12]:


df_train.describe()


# In[13]:


df_train['Item_Weight'].describe()


# In[14]:


df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(),inplace=True)


# In[15]:


df_train.isnull().sum()


# In[16]:


df_train['Outlet_Size']


# In[17]:


df_train['Outlet_Size'].value_counts()


# In[18]:


df_train['Outlet_Size'].mode()


# In[19]:


df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0],inplace=True)
df_test['Outlet_Size'].fillna(df_test['Outlet_Size'].mode()[0],inplace=True)


# In[20]:


df_train


# In[21]:


df_train.isnull().sum()


# In[22]:


df_test.isnull().sum()


# In[23]:


#Selecting features based on general Requirements 


# In[24]:


df_train.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)
df_test.drop(['Item_Identifier','Outlet_Identifier'],axis=1,inplace=True)


# In[25]:


df_train


# In[ ]:





# In[ ]:





# In[26]:


#EDA using dtale library


# In[27]:


#conda install dtale -c conda-forge


# In[28]:


#conda install -c plotly python-kaleido


# In[29]:


#!pip install dtale


# In[30]:


#import dtale


# In[31]:


#dtale.show(df_train)


# In[ ]:





# In[ ]:





# In[32]:


#EDA Using pandas profiling


# In[33]:


get_ipython().system(' pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip ')


# In[34]:


import pandas_profiling
#from pandas_profiling import ProfileReport


# In[35]:


profile= df_train.profile_report(title="Pandas Profiling Report")


# In[36]:


profile


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


#EDA using klib library


# In[38]:


get_ipython().system('pip install klib')
import klib


# In[39]:


klib.cat_plot(df_train)


# In[40]:


klib.corr_mat(df_train)


# In[41]:


klib.corr_plot(df_train)


# In[42]:


klib.dist_plot(df_train)


# In[43]:


klib.missingval_plot(df_train)


# In[44]:


klib.data_cleaning(df_train)


# In[45]:


klib.clean_column_names(df_train)


# In[46]:


df_train.info()


# In[47]:


df_train=klib.convert_datatypes(df_train)


# In[48]:


df_train.info()


# In[49]:


klib.drop_missing(df_train)


# In[50]:


klib.mv_col_handling(df_train)


# In[51]:


klib.pool_duplicate_subsets(df_train)


# In[ ]:





# In[ ]:





# In[52]:


#Preprocessing task before model building


# In[53]:


# 1) Label encoding


# In[103]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[104]:


df_train['item_fat_content']= le.fit_transform(df_train['item_fat_content'])
df_train['item_type']= le.fit_transform(df_train['item_type'])
df_train['outlet_size']= le.fit_transform(df_train['outlet_size'])
df_train['outlet_location_type']= le.fit_transform(df_train['outlet_location_type'])
df_train['outlet_type']= le.fit_transform(df_train['outlet_type'])


# In[105]:


df_train.head()


# In[57]:


# 2) One hot Encoding


# In[58]:


#df_train = pd.get_dummies(df_train, columns=['item_fat_content','outlet_size','outlet_location_type','outlet_type'])


# In[59]:


#df_train


# In[ ]:





# In[60]:


# Splitting our data into train and test


# In[61]:


x=df_train.drop('item_outlet_sales',axis=1)


# In[62]:


y=df_train['item_outlet_sales']


# In[63]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=101,test_size=0.2)


# In[64]:


x_train


# In[65]:


x_test


# In[66]:


x.describe()


# In[67]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[68]:


x_train_std= sc.fit_transform(x_train)


# In[69]:


x_test_std=sc.transform(x_test)


# In[70]:


x_train_std


# In[71]:


x_test_std


# In[72]:


y_train


# In[73]:


y_test


# In[74]:


import joblib


# In[77]:


joblib.dump(sc,r'C:\Users\HP-PC\Documents\Big Mart Sales\models\sc.sav')


# In[ ]:





# In[ ]:





# In[78]:


# Model building


# #Linear regression

# In[79]:


from sklearn.linear_model import LinearRegression 
lr= LinearRegression() 


# In[80]:


lr.fit(x_train_std,y_train)


# In[81]:


x_test.head()


# In[82]:


y_pred_lr=lr.predict(x_test_std)


# In[83]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[84]:


print(r2_score(y_test,y_pred_lr))
print(mean_absolute_error(y_test,y_pred_lr))
print(np.sqrt(mean_squared_error(y_test,y_pred_lr)))


# In[85]:


joblib.dump(lr,r'C:\Users\HP-PC\Documents\Big Mart Sales\models\lr.sav')


# # Random forest regressor

# In[86]:


from sklearn.ensemble import RandomForestRegressor
rf= RandomForestRegressor(n_estimators=1000)


# In[87]:


rf.fit(x_train_std,y_train)


# In[88]:


y_pred_rf=rf.predict(x_test_std)


# In[89]:


print(r2_score(y_test,y_pred_rf))
print(mean_absolute_error(y_test,y_pred_rf))
print(np.sqrt(mean_squared_error(y_test,y_pred_rf)))


# In[92]:


joblib.dump(rf,r'C:\Users\HP-PC\Documents\Big Mart Sales\models\rf.sav')


# In[ ]:





# In[93]:


#Hyper parameter tuning


# In[94]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = RandomForestRegressor()
n_estimators = [10, 100, 1000]
max_depth=range(1,31)
min_samples_leaf=np.linspace(0.1, 1.0)
max_features=["auto", "sqrt", "log2"]
min_samples_split=np.linspace(0.1, 1.0, 10)

# define grid search
grid = dict(n_estimators=n_estimators, max_depth=max_depth)

#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=101)

grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           scoring='r2',error_score=0,verbose=2,cv=2)

grid_search_forest.fit(x_train_std, y_train)

# summarize results
print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
means = grid_search_forest.cv_results_['mean_test_score']
stds = grid_search_forest.cv_results_['std_test_score']
params = grid_search_forest.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


# In[95]:


grid_search_forest.best_params_


# In[96]:


grid_search_forest.best_score_


# In[97]:


y_pred_rf_grid=grid_search_forest.predict(x_test_std)


# In[98]:


r2_score(y_test,y_pred_rf_grid)


# In[99]:


# Save your model


# In[100]:


import joblib


# In[101]:


joblib.dump(grid_search_forest,r'C:\Users\HP-PC\Documents\Big Mart Sales\models\random_forest_grid.sav')


# In[ ]:




