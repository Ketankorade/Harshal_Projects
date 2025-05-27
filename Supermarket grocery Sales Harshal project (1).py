#!/usr/bin/env python
# coding: utf-8

# # Supermarket grocery Sales 

# # 1 Import Requried Liabraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


# Laod the Dataset
data = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")
data.head


# # 2 Basic Data Exploration

# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.columns


# In[8]:


# check for null values
data.isnull().sum()


# In[9]:


# Unique values in each column
data.nunique()


# # 3 Data cleaning

# In[10]:


# check for duplicate
data.duplicated().sum()


# In[11]:


# checking missing values
data.isnull().sum().sum()


# In[12]:


# Convert 'Order Date' to datetime format
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d-%m-%Y', errors='coerce')


# In[15]:


data.head(10)


# # 4 Feature Engineering

# In[16]:


# Extracting date parts
data['Year'] = data['Order Date'].dt.year
data['Month']=  data['Order Date'].dt.month
data['Day']= data['Order Date'].dt.day
data['Weekday'] = data['Order Date'].dt.day_name()

# Revenues
data['Revenue'] = data['Sales']

# priview enhanced dataset
print(data[['Order Date','Year','Month','Day','Weekday','Revenue']].head())


# # 5 Exploratory data analysis

# In[17]:


monthly_sales = data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Revenue', data=monthly_sales)
plt.title('Monthly Sales Trend')
plt.ylabel('Total Revenue')
plt.xlabel('Date')
plt.show()


# In[18]:


# Top ten sub-categories by revenue
top_subcats = data.groupby('Sub Category')['Revenue'].sum().sort_values(ascending=False).head(10)
top_subcats.plot(kind='bar',figsize=(10,5),title='Top 10 Sub_Categories by Revenue')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()


# In[19]:


# Monthly Sales Trend
monthly_sales = data.groupby(['Year', 'Month'])['Revenue'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))

plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Revenue', data=monthly_sales)
plt.title('Monthly Sales Trend')
plt.ylabel('Total Revenue')
plt.xlabel('Date')
plt.show()


# In[20]:


# 3. Correlation Heatmap
corr = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
plt.title('Correlation Heatmap')
plt.show()


# # 6 Feature Engineering and Model Building

# In[21]:


# Encode categorical variables
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])
data['Sub Category'] = label_encoder.fit_transform(data['Sub Category'])
data['City']= label_encoder.fit_transform(data['City'])
data['Region']= label_encoder.fit_transform(data['Region'])
data['State']= label_encoder.fit_transform(data['State'])


# In[22]:


# Feature selection
features = data[['Category','Sub Category','City','Region','State','Discount','Month','Year']]
target = data['Sales']


# In[23]:


# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=42)


# In[27]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# In[30]:


# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[33]:


# Model Building - Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[34]:


# Model Evalution
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
print(f'mean Squared Error:{mse}')
print(f'R-squared:{r2}')


# # Conclusion
# This analysis helps understand the sales trends, category performance, and the impact of discounts on profit.
# 

# In[ ]:




