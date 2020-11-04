#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Retail analysis: Build a Recency Frequency Monetary Model 
# Perform customer segmentation with KMeans Clustering
# Create segments to determine total customer value for the retail outlets

# Credentials - kasham1991@gmail.com / Karan Sharma


# In[2]:


# Importing the basic libraries

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(color_codes = True)
sns.set(style = 'white')


# In[3]:


# Loading the datasets

train = pd.read_excel('C:/Datasets/retail_train.xlsx')
test = pd.read_excel('C:/Datasets/retail_test.xlsx')


# In[4]:


# Looking at the training and test datasets

print("Shape of train data")
print("-------------------")
print(train.shape)
print("-------------------")
print("Columns of train data")
print("---------------------")
print(list(train.columns))
print("-------------------")
print("Types of train columns")
print("----------------------")
print(train.info())
print("-------------------")
print("Shape of test data")
print("-------------------")
print(test.shape)
print("-------------------")
print("Columns of test data")
print("---------------------")
print(list(test.columns))
print("-------------------")
print("Types of test columns")
print("----------------------")
print(test.info())


# In[5]:


# Basic statiscal analysis
# There are multiple NaN values
# How can quantity and unit price have negative values?

train.describe(include = 'all')
#test.describe(include = 'all')
#train.head(5)
#test.head(5)


# In[6]:


# Looking for missing values aong with NaN
# There is a significant portion of missiong values in the customer id and description
# it is beter to drop them completely
print(train.isnull().sum())
print(test.isnull().sum())


# In[7]:


# Dropping null values for customer id since it is not an important categorical data
# Since description is discrete, we can use mode to replace the null values

train.dropna(subset = ['CustomerID'], inplace = True)
test.dropna(subset = ['CustomerID'], inplace = True)

train['Description'].fillna(train['Description'].mode()[0], inplace = True)
test['Description'].fillna(test['Description'].mode()[0], inplace = True)

#train.isnull().sum()
#test.isnull().sum()


# In[8]:


# Looking for duplicate records
# 2656 duplicate rows/468 duplicate rows

print(train[train.duplicated()])
print(test[test.duplicated()])


# In[9]:


# Dropping duplicate rows
train.drop_duplicates(inplace = True)
test.drop_duplicates(inplace = True)

#train[train.duplicated()]
#test[test.duplicated()]


# In[10]:


# Exploratory Data Analysis
# Plotting the count of the dtypes

train.dtypes
#test.dtypes

train.dtypes.value_counts().plot(kind = 'barh', color = 'skyblue')
plt.title("Dtype type counts in Train set", fontsize = 12)


# In[11]:


test.dtypes.value_counts().plot(kind = 'barh', color = 'purple')
plt.title("Dtype counts in Test set", fontsize = 12)


# In[12]:


# Looking for outliers with the box plot method
# Quantity and unitprice have a few outliers

train.boxplot()
plt.title("Boxplot for Train set", fontsize = 12)


# In[13]:


test.boxplot()
plt.title("Boxplot for test set", fontsize = 12)


# In[14]:


# Performing cohort analysis
# A cohort is a group of subjects that share a defining characteristic
# Create month cohorts and analyze active customers for each cohort
# Analyze the retention rate of customers


# In[15]:


# Creating monthly cohorts on the basis of customer id and invoice date
# Converting cust id to integer from float

train['CustomerID'] = train['CustomerID'].apply(lambda x: int(x))
test['CustomerID'] = test['CustomerID'].apply(lambda x: int(x))


# In[16]:


# Looking for unique values in invoice date on the basis of years and months

print(train['InvoiceDate'].nunique())
print(train['InvoiceDate'].dt.year.unique())
print(train['InvoiceDate'].dt.month.unique())


# In[17]:


# Using the above to extract string from time
# Looking at the no of unique months 

train['InvoiceMonth'] = train['InvoiceDate'].apply(lambda x: x.strftime('%Y-%m'))
test['InvoiceMonth'] = test['InvoiceDate'].apply(lambda x: x.strftime('%Y-%m'))

train['InvoiceMonth'].unique()
#test['InvoiceMonth'].unique()


# In[18]:


# Plotting the customer count and invoice count across the unique months
# Here we are grouping invoice month on the sum of cust id and invoice no
# We can see that the frequency of the transactions keep increasing over time

count_group = train.groupby('InvoiceMonth').agg({'CustomerID':pd.Series.nunique,
                                        'InvoiceNo':pd.Series.nunique
                                         })
count_group.columns = ['Customercount','Invoice count']
count_group.plot(figsize = (12, 6))


# In[19]:


# Analysis of retention rate; how many customers continue to purchase
# Cohort month is the time from which the user starts to purchase from the website
# Lets look at the min cohort month as this becomes the initial month of pruchase
# Grouping cohort month by customer id and invoice month

train['CohortMonth'] = train.groupby('CustomerID')['InvoiceMonth'].transform('min')
test['CohortMonth'] = test.groupby('CustomerID')['InvoiceMonth'].transform('min')

train.head()
#test.head()


# In[20]:


# Creating a function for deriving the cohort index 
# It is the difference between the invoice month and cohort month
# It will tell us the time lapse between a specific transaction and the first transaction made by the user; user retention

def get_date(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month

train['InvoiceMonth'] = pd.to_datetime(train['InvoiceMonth'])
train['CohortMonth'] = pd.to_datetime(train['CohortMonth'])
test['InvoiceMonth'] = pd.to_datetime(test['InvoiceMonth'])
test['CohortMonth'] = pd.to_datetime(test['CohortMonth'])

invoice_year, invoice_month = get_date(train, 'InvoiceMonth')
cohort_year, cohortmonth = get_date(train, 'CohortMonth')
year_diff = invoice_year - cohort_year
month_diff = invoice_month - cohortmonth
train['CohortIndex'] = year_diff * 12 + month_diff 


invoice_year, invoice_month = get_date(test, 'InvoiceMonth')
cohort_year, cohortmonth = get_date(test, 'CohortMonth')
year_diff = invoice_year - cohort_year
month_diff = invoice_month - cohortmonth
test['CohortIndex'] = year_diff * 12 + month_diff 

train.head()
#test.head()


# In[21]:


# Creating a cohort table with index
# Grouping by month, index and id
# Looking at the count

cohort_data = train.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].apply(pd.Series.nunique).reset_index()

cohort_count = cohort_data.pivot_table(index = 'CohortMonth',
                                       columns = 'CohortIndex',
                                       values = 'CustomerID')
cohort_count


# In[22]:


# Plotting the above table
# 248/929 users were purchasing online post 2010-12-01, after that there is a decline in the customer base gradually

cohort_count.plot(figsize = (20, 7))
plt.show()


# In[23]:


# As seen from above, the retention rate is low
# Lets plot a heatmap for the same
# Rounding off the float values by 100
cohort_size = cohort_count.iloc[:, 0]
retention = cohort_count.divide(cohort_size, axis = 0)
retention.round(3) * 100


# In[24]:


# Using the percentage values since retention is a rate
# 40% is the highest retention rate observed

plt.figure(figsize = (11, 9))
plt.title('Retention Rate Per Cohort')
sns.heatmap(data = retention, 
            annot = True, 
            fmt = '.0%',
            cmap = 'gist_stern')
plt.show()


# In[25]:


# Calculating the average quantity purchased by cohort
# Grouping by mean quantity; month & index
cohort_data2 = train.groupby(['CohortMonth', 'CohortIndex'])['Quantity'].mean().reset_index()

average_qty = cohort_data2.pivot_table(index = "CohortMonth",
                        columns = "CohortIndex",
                        values = "Quantity").round(1)
average_qty


# In[26]:


# Plotting the heatmap for the above
plt.figure(figsize = (11,9))
plt.title('Mean Average Quantity Purchased Per Cohort')
sns.heatmap(data = average_qty, 
            annot = True, 
            cmap = "gist_stern")
plt.show()


# In[27]:


# Building the Recency Frequency Monetary Parameters 
# Recency means the number of days since a customer made the last purchase
# Frequency is the number of purchase in a given period
# It could be 3 months, 6 months or 1 year. Monetary is the total amount of money a customer spent in that given period
# Therefore, big spenders will be differentiated among other customers such as MVP (Minimum Viable Product) or VIP


# In[28]:


# Calculating the total sales
# Making the relevant columns for RFM
train['Sales'] = train['Quantity'] * train['UnitPrice']
test['Sales'] = test['Quantity'] * test['UnitPrice']

train[['InvoiceNo','InvoiceDate','CustomerID','Sales']].head()


# In[29]:


# Recency has to claculated from a specific date
# Using the time delta function; difference between two dates

from datetime import timedelta
tym = train['InvoiceDate'].max() + timedelta(days = 1)
tym


# In[30]:


# Grouping by cust id
# Creating a new table
rfm = train.groupby('CustomerID').agg({
                                        'InvoiceDate' : lambda x: (tym-x.max()).days,
                                         'InvoiceNo'  : lambda x: len(x),
                                         'Sales' : lambda x : sum(x)
                                       })
rfm.rename(columns = {'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'Sales': 'MonetaryValue'}, inplace = True)

rfm.head()


# In[31]:


# Give recency, frequency, and monetary scores individually by dividing them into quartiles
# Combine three ratings to get a RFM segment (as strings)
# Get the RFM score by adding up the three ratings
# Analyze the RFM segments by summarizing them and comment on the findings

# Rate “recency" for customer who has been active more recently higher than the less recent customer, 
# because each company wants its customers to be recent
# Rate “frequency" and “monetary" higher, because the company wants the customer to visit more often and spend more money


# In[32]:


# Calculating RFM groups,labels and quartiles with the qcut function  

r_labels = range(4, 0, -1)
f_labels = range(1, 5)
m_labels = range(1, 5)

# Assign these labels to 4 equal percentile groups 
r_groups = pd.qcut(rfm['Recency'], q = 4, labels = r_labels)
# Assign these labels to 4 equal percentile groups 
f_groups = pd.qcut(rfm['Frequency'], q = 4, labels = f_labels)
# Assign these labels to three equal percentile groups 
m_groups = pd.qcut(rfm['MonetaryValue'], q = 4, labels = m_labels)


# In[33]:


print(r_groups[:5])
print(f_groups[:5])
print(m_groups[:5])


# In[34]:


# Adding the new columns to original rmf
rfm = rfm.assign(R = r_groups.values, F = f_groups.values, M = m_groups.values)
rfm.head()


# In[35]:


# Combine three ratings to get a RFM segment (as strings)
rfm['RFM_segment'] = rfm.apply(lambda x: (str(x['R']) + str(x['F']) + str(x['M'])), axis=1)
rfm.head()


# In[36]:


# Get the RFM score by adding up the three ratings.
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis = 1)
rfm['RFM_Score'] = rfm['RFM_Score'].apply(lambda x : int(x))
rfm.head()


# In[37]:


# No of unique segments
# Looking at the top customers under the RFM segment of 444

print(rfm['RFM_segment'].nunique())
print(rfm['RFM_Score'].unique())
rfm[rfm['RFM_segment'] == '444'].head()


# In[38]:


# Creating a function to define rfm_level function on the basis of importance

def rfm_level(df):
    if df['RFM_Score'] >= 9:
        return 'Important'
    elif ((df['RFM_Score'] >= 8) and (df['RFM_Score'] < 9)):
        return 'Good'
    elif ((df['RFM_Score'] >= 7) and (df['RFM_Score'] < 8)):
        return 'Okay'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7)):
        return 'Neutral'
    elif ((df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6)):
        return 'Might'
    elif ((df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5)):
        return 'Needs Attention'
    else:
        return 'Activate'
# Create a new variable RFM_Level
rfm['RFM_Level'] = rfm.apply(rfm_level, axis=1)
rfm.head()


# In[39]:


# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = rfm.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)

rfm_level_agg


# In[40]:


# Plotting the above information
# Most of the customers 1600 plus are in the important RFM_segment

rfm['RFM_Level'].value_counts().plot(kind = 'barh', figsize = (10, 5), color = 'skyblue')
plt.title('Customer Distribution Across Different RFM levels', fontsize = 15)
plt.show()


# In[41]:


# Data Modeling with Kmeans Clustering
# Standardize the retail data

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[42]:


cols = ['CustomerID','Sales']
x_train = train[cols]

sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)


# In[43]:


# Looking for the optimum no of clusters/cluster sum of squares (WCSS)
# Sum of squared distances of samples to their closest cluster centre

wcss = []

for i in range(1, 12):
    
    kmeans = KMeans(n_clusters = i, 
                    init       = 'k-means++', 
                    max_iter   = 300, 
                    n_init     = 10, 
                    random_state = 0)
    
    kmeans.fit(x_train_scaled)
    
    wcss.append(kmeans.inertia_)


# In[44]:


# Plotting the results into a line graph 
# Creating an elbow chart
# WCSS - Within cluster sum of squares
# Optimum no of clusters is 4 as per the elbow in the plot

plt.plot(range(1, 12), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# In[45]:


# Fitting the data to the graph
# Yellowbrick is a suite of visual analysis and diagnostic tools designed to facilitate machine learning with scikit-learn

from yellowbrick.cluster import KElbowVisualizer
visualizer = KElbowVisualizer(kmeans, k = (1, 12))
visualizer.fit(x_train_scaled)    
visualizer.show()   
#visualizer.poof()   


# In[46]:


# Repeating the above steps for test data

x_test = test[cols]
x_test_scaled = sc.fit_transform(x_test)


# In[47]:


# Applying kmeans

kmeans = KMeans(n_clusters = 4,  # optimum cluster
                    init       = 'k-means++', 
                    max_iter   = 300, 
                    n_init     = 10, 
                    random_state = 0)

y_pred = kmeans.fit_predict(x_test_scaled)


# In[48]:


# Creating a new cloumn for y_pred
# Making predictions on the training set

test['Cluster'] = y_pred
y_pred_train = kmeans.fit_predict(x_train_scaled)
train['Cluster'] = y_pred_train
#train


# In[49]:


# No of Clusters along with the no of customers in it
# We were able to build a model that can classify new customers into "low value" , "middle value" and "high value" groups

test['Cluster'].unique()
test['Cluster'].value_counts()


# In[50]:


# Thank You :)

