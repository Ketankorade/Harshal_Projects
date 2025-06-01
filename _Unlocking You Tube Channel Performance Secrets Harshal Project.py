#!/usr/bin/env python
# coding: utf-8

# # Unlocking You Tube Channel Performance Secrets

# # 1 Import Liabraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta


# # 2 Load the Dataset

# In[2]:


# Load the dataset
data = pd.read_csv("youtube_channel_real_performance_analytics.csv")
data


# In[3]:


data.head()


# # 3 Data Cleaning & Preprocessing

# In[4]:


# Fill or drop null values
data = data.dropna() # Drop rows with missing values 


# In[5]:


get_ipython().system('pip install isodate')


# In[6]:


import pandas as pd
import isodate
import pandas as pd
import isodate

data = pd.DataFrame({'Video Duration': ['PT1H2M30S', 'PT45M', 'PT30S', None]})

data['Video Duration'] = data['Video Duration'].dropna().apply(
    lambda x: isodate.parse_duration(x).total_seconds() if pd.notna(x) else None)
 


# In[7]:


# Drop missing rows for simplicity
data.dropna(inplace=True)

# Convert 'Video Duration'= data['Video Duration'].apply(lambdax:isodate.parse_duration(x).total_seconds())


# # 4 Exploratory Data Analysis

# In[9]:


# Reload the original dataset
data = pd.read_csv("youtube_channel_real_performance_analytics.csv")

# Fill or drop null values in the original dataset
data = data.dropna()  # Drop rows with missing values 
# Pairplot to visualize relationships
sns.pairplot(data[['Revenue per 1000 Views (USD)', 'Views','Subscribers', 'Estimated Revenue (USD)']])
plt.show()


# In[10]:


# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data[['Revenue per 1000 Views (USD)', 'Views','Subscribers', 'Estimated Revenue (USD)']].corr(),annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[11]:


# Top Performers by Revenue
top_videos = data.sort_values(by='Estimated Revenue (USD)', ascending=False).head(10)
display(top_videos[['ID', 'Estimated Revenue (USD)', 'Views', 'Subscribers']])


# # 5 Feature Engineering

# In[12]:


import numpy as np

# Avoid division by zero
data['Revenue per View'] = np.where(data['Views'] > 0, data['Estimated Revenue (USD)'] / data['Views'], 0)

# Check if 'Comments' column exists, if not, create it with 0 values
if 'Comments' not in data.columns:
    data['Comments'] = 0  
# Now calculate Engagement Rate
data['Engagement Rate'] = np.where(data['Views'] > 0,
(data['Likes'] + data['Shares'] + data['Comments']) / data['Views'] * 100, 0)


# # 6 Data Visualization

# In[13]:


# Revenue Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['Estimated Revenue (USD)'], bins=50, kde=True, color='green')
plt.title("Revenue Distribution")
plt.xlabel("Revenue (USD)")
plt.ylabel("Frequency")
plt.show()


# In[14]:


# Revenue vs Views
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Views'], y=data['Estimated Revenue (USD)'])
plt.title("Revenue vs Views")
plt.xlabel("Views")
plt.ylabel("Revenue (USD)")
plt.show()


# # 7 Build Predective Model

# In[15]:


# Define features and target
features = ['Views', 'Subscribers', 'Likes', 'Shares', 'Comments', 'Engagement Rate']
X = data[features]
y = data['Estimated Revenue (USD)']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# # 8 Evalute Model

# In[16]:


# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# # 9 Insight & Export Model

# In[17]:


# Feature Importance
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df)
plt.title("Feature Importance")
plt.show()


# In[ ]:




