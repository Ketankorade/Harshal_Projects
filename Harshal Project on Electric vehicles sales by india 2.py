#!/usr/bin/env python
# coding: utf-8

# # Electric Vehicle Sales by State in India

# # step 1 : Import the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# # step 2 : Data Colllection

# In[2]:


EV_sales = pd.read_csv("Electric Vehicle Sales by State in India.csv")
EV_sales


# In[3]:


# Display the first few rows of the data set
print(EV_sales.head())


# In[4]:


print(EV_sales.info())


# In[5]:


EV_sales.sample(30)


# In[6]:


EV_sales.describe()


# In[7]:


# Checking Duplicate value
EV_sales.duplicated().sum()


# In[8]:


EV_sales.shape


# # 3 Data Processing

# In[9]:


# Handle the  missing values
print(EV_sales.isnull().sum())


# In[10]:


# drop rows where EV_sales quantity or data missing
EV_sales = EV_sales.dropna(subset=['EV_Sales_Quantity','Date'])


# In[11]:


# Convert 'Date'to detemine format
EV_sales['Date'] = pd.to_datetime(EV_sales['Date'], errors='coerce',dayfirst=True)


# In[12]:


# Drop the row with invalid dates
EV_sales = EV_sales.dropna(subset=['Date'])


# In[13]:


# standardize string columns
categorical_cols = ['State','Vehicle_Class','Vehicle_Category','Vehicle_Type',]
for col in categorical_cols:
    EV_sales[col] = EV_sales[col].astype(str).str.strip().str.title()
    


# In[14]:


# check the clear data 
print("\nAfter cleaning:")
print(EV_sales[categorical_cols + ['Date','EV_Sales_Quantity']].head())


# In[15]:


# we clean the values in key columns
# convert data to proper datatime format


# # 4 Exploratory data Analysis

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


# Create time-based features
EV_sales['Year'] = EV_sales['Date'].dt.year
EV_sales['Month'] = EV_sales['Date'].dt.month
EV_sales['Quarter'] = EV_sales['Date'].dt.quarter


# In[18]:


# EV_sales trend over the years
sales_by_year =EV_sales.groupby('Year')['EV_Sales_Quantity'].sum().reset_index()
sns.lineplot(data=sales_by_year, x='Year',y='EV_Sales_Quantity',marker='o')
plt.title('EV Sales Trend Over the Years')
plt.ylabel('Total EV Sales')
plt.xlabel('Year')
plt.show()


# In[19]:


# top ten states by total EV sales
top_states = EV_sales.groupby('State')['EV_Sales_Quantity'].sum().sort_values(ascending=False).head(10)
top_states.plot(kind='barh',color= 'skyblue')
plt.title('Top 10 states by Total EV sales')
plt.xlabel('EV_Sales_Quantity')
plt.gca().invert_yaxis
plt.show()


# In[20]:


# EV sales by vehicle type
sns.barplot(data=EV_sales, x = 'Vehicle_Type', y ='EV_Sales_Quantity', estimator=sum,ci=None,palette='muted')
plt.title('Total EV sales by Vehicle type')
plt.ylabel('EV_Sales_Quantity')
plt.xticks(rotation=45)
plt.show()


# In[21]:


# EV_sales by vehicle category (passenger/commercial)
sns.boxplot(data=EV_sales, x='Vehicle_Category', y = 'EV_Sales_Quantity')
plt.title('Distribution of EV sales by vehicle category')
plt.ylabel('EV_Sales Quantity')
plt.xticks(rotation=45)
plt.show()


# In[22]:


# above information we can observed that EV sales have change over time
# which state lead in EV adoption
# which vehicle is most popular


# # 5 Feature Engineering

# In[23]:


from sklearn.preprocessing import LabelEncoder

# Create more time-based features
EV_sales['Day'] = EV_sales['Date'].dt.day
EV_sales['Weekday'] = EV_sales['Date'].dt.weekday 

# Encode categorical features
label_encoders = {}
categorical_features = ['State', 'Vehicle_Class', 'Vehicle_Category', 'Vehicle_Type']

for col in categorical_features:
    le = LabelEncoder()
    EV_sales[col + '_Code'] = le.fit_transform(EV_sales[col])
    label_encoders[col] = le  

# Select final features and target
features = [
    'Year', 'Month', 'Quarter', 'Day', 'Weekday',
    'State_Code', 'Vehicle_Class_Code',
    'Vehicle_Category_Code', 'Vehicle_Type_Code'
]

X = EV_sales[features]
y = EV_sales['EV_Sales_Quantity']

# Preview final dataset for modeling
print("Features ready for modeling:")
print(X.head())
print("\nTarget variable:")
print(y.head())


# # 6 Modelling

# In[24]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# split the dataset
x_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Initialize & train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# predict on data set
y_pred = model.predict(X_test)

# preview prediction
results_EV_sales = pd.DataFrame({'Actual': y_test.values,'Predicted': y_pred})
print("Sample Predictions:")
print(results_EV_sales.head())


# # 7 Model_evalution

# In[25]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np


# In[26]:


# Calculate evalution metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)


# In[27]:


# print results
print("Model Evolution Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error(RSME):{rmse:.2f}")
print(f"R^2 score:{r2:.2f}")


# # 8 Visualization of model Results

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


# plot actual vs predicted values
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test,y=y_pred,alpha=0.6)
plt.plot([y_test.min(), y_test.max()],[y_test.min(),y_test.max()],'r--', lw=2)
plt.xlabel('Actual EV Sales')
plt.ylabel('Predicted EV Sales')
plt.title('Actual vs Predicted EV Sales')
plt.grid(True)
plt.show()


# In[30]:


# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)


# In[31]:


# plot feature importance
plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature',palette='viridis')
plt.title('Feature Importance in EV Sales Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# # Conclusion
# The analysis of EV sales in India shows a strong growth trend, with certain states leading in adoption. Two-wheeler and three-wheeler EVs dominate the market. Seasonal variations and policy impacts influence sales. Continuous growth suggests increasing EV acceptance, but infrastructure improvements are crucial for sustained expansion.
# 
# 

# In[ ]:




