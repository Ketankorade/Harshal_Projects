#!/usr/bin/env python
# coding: utf-8

# # Objective
# Analyzing e-commerce furniture data helps identify sales trends, customer preferences, and inventory demands. It optimizes pricing, marketing strategies, and supply chain efficiency while enhancing user experiences. This drives better decision-making, boosts revenue, and fosters customer satisfaction in competitive markets.

# # Importing necessary libraries¶

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# # Loading data

# In[3]:


df = pd.read_csv("ecommerce_furniture_dataset_2024 (1).csv")


# In[4]:


df


# In[5]:


#head method 
df.head()


# In[6]:


# tail method 
df.tail()


# # Data processing

# # Now we will check shape, dimension , columns 

# In[7]:


df.shape


# In[8]:


df.ndim


# In[9]:


df.columns


# In[10]:


df.dtypes


# In[11]:


df.info()


# In[12]:


df['tagText'].nunique()


# In[13]:


df['tagText'].value_counts()


# # Handling missing values 

# In[14]:


df.isnull().sum()


# In[15]:


df.isna().sum()


# In[16]:


df.fillna(0,inplace=True)


# In[17]:


df.isnull().sum()


# In[18]:


df.describe()


# In[19]:


df.dtypes


# In[ ]:


# Now we will convert object dtype of original price and price in float64 & tagText into category


# In[20]:


df['originalPrice']=df['originalPrice'].str.replace('$','').str.replace(',','')


# In[21]:


df['price']=df['price'].str.replace('$','').str.replace(',','')


# In[22]:


df['originalPrice']=pd.to_numeric(df['originalPrice'])


# In[23]:


df['price']=pd.to_numeric(df['price'])


# In[24]:


df.dtypes


# In[25]:


df


# In[26]:


df['tagText']=df['tagText'].astype('category').cat.codes


# In[27]:


df.fillna(0,inplace=True)


# In[28]:


df


# # Exploratory Data Analysis
# Now that our data is clean, let's perform some exploratory data analysis to understand the distribution and relationship with data 

# In[29]:


# Summary statistics
df.describe()


# In[30]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[31]:


# Distribution of prices
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[32]:


# Relationship between price and number of items sold
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='sold', data=df)
plt.title('Price vs. Number of Items Sold')
plt.xlabel('Price')
plt.ylabel('Number of Items Sold')
plt.show()


# In[33]:


# for original price feature 
plt.boxplot(df['originalPrice'])
plt.show()

# As it is not showing the 25 and 75 quartile of range
# In[34]:


# for price feature
plt.boxplot(df['price'])
plt.show()

# its nicely showing pertfect quartile 
# # Visualization

# In[35]:


plt.figure(figsize=(20,8))
df.groupby('productTitle')["originalPrice"].sum().sort_values(ascending=False).head(10).plot(kind='bar',color='green')
plt.title('Top 10 Product by its originalPrice',size=35)
plt.ylabel('originalPrice($)')
plt.show()

# Garden Furniture Set 7 PCS, Garden Fire Pit Table Patio Sets, No-Slip Cushions and Waterproof Covers, Garden Furniture Set' is the most expensive product in terms of original price.
# In[36]:


plt.figure(figsize=(20,8))
df.groupby('productTitle')["price"].sum().sort_values(ascending=False).head(10).plot(kind='bar',color='cyan')
plt.title('Top 10 Product by its Price',size=35)
plt.ylabel('Price($)')
plt.show()

Garden Furniture Set 7 PCS, Garden Fire Pit Table Patio Sets, No-Slip Cushions and Waterproof Covers, Garden Furniture Set' is the most expensive product in terms of price
# In[38]:


plt.figure(figsize=(20,8))
df.groupby('productTitle')["sold"].sum().sort_values(ascending=False).head(10).plot(kind='bar',color='coral')
plt.title('Top 10 Product by its Sold Quantity',size=35)
plt.ylabel('Sold Q')
plt.show()

Portable round Folding Chair Accordion Chair Height Adjustment Simple Tool Elephant Swing Playground Queue Chair is the most solded product in quantity
# # Conclusion
# In this starter-code notebook, we have loaded, cleaned, and performed exploratory data analysis on the E-commerce Furniture Dataset for 2024. We have visualized the distribution of prices and examined the relationship between price and the number of items sold. Further analysis can be conducted to uncover more insights and trends.
# 
# 

# In[ ]:




