
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# In[2]:


import os
os.chdir(r"E:\MY\Project\Cab Fare Prediction")


# In[3]:


Train = pd.read_csv("train.csv")


# In[4]:


#Train.head()
Train.shape
#16067


# In[5]:


Test =  pd.read_csv("test.csv")


# In[6]:


#Test.head()
Test.shape
#(9914, 6)


# In[7]:


#Train.describe()
Train.dtypes


# In[8]:


Train["fare_amount"] = pd.to_numeric(Train["fare_amount"],errors='coerce')


# In[9]:


def missin_val(df):
    missin_val = pd.DataFrame(df.isnull().sum())
    missin_val = missin_val.reset_index()
    missin_val = missin_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})
    missin_val['Missing_percentage'] = (missin_val['Missing_percentage']/len(df))*100
    missin_val = missin_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)
    return(missin_val)

print("The missing value percentage in training data : \n\n",missin_val(Train))
print("\n")
print("The missing value percentage in test data : \n\n",missin_val(Test))
print("\n")

#Impute the missing values
Train["passenger_count"] = Train["passenger_count"].fillna(Train["passenger_count"].median())
Train["fare_amount"] = Train["fare_amount"].fillna(Train["fare_amount"].median())

#check if any missing  value still exists
print("Is there still any missing value in the training data:\n\n",Train.isnull().sum())
print("\n")


# In[10]:


#Split our Datetime into individual columns for ease of data processing and modelling
def align_datetime(df):
    df["pickup_datetime"] = df["pickup_datetime"].map(lambda x: str(x)[:-3])
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], format='%Y-%m-%d %H:%M:%S')
    df['year'] = df.pickup_datetime.dt.year
    df['month'] = df.pickup_datetime.dt.month
    df['day'] = df.pickup_datetime.dt.day
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    return(df["pickup_datetime"].head())
    
align_datetime(Train)
align_datetime(Test)


# In[11]:


#Remove the datetime column
Train.drop('pickup_datetime', axis=1, inplace=True)
Test.drop('pickup_datetime', axis=1, inplace=True)


# In[12]:


#Checking NA in the fresh Dataset
Train.isnull().sum()
Train=Train.fillna(Train.mean())
Train.isnull().sum()


# In[13]:


#Setting proper data type for each columns
Train= Train.astype({"fare_amount":float,"pickup_longitude":float,"pickup_latitude":float,"dropoff_longitude":float,"dropoff_latitude":float,"passenger_count":int,"year":int,"month":int ,"day" :int,"weekday":int,"hour":int})
Train.dtypes
Test = Test.astype({"pickup_longitude":float,"pickup_latitude":float,"dropoff_longitude":float,"dropoff_latitude":float,"passenger_count":int,"year":int,"month":int ,"day" :int,"weekday":int,"hour":int})
Test.dtypes


# In[14]:


def proper_data(df):
    df = df[((df['pickup_longitude'] > -78) & (df['pickup_longitude'] < -70)) & 
           ((df['dropoff_longitude'] > -78) & (df['dropoff_longitude'] < -70)) & 
           ((df['pickup_latitude'] > 37) & (df['pickup_latitude'] < 45)) & 
           ((df['dropoff_latitude'] > 37) & (df['dropoff_latitude'] < 45)) & 
           ((df['passenger_count'] > 0) & (df['passenger_count'] < 7)) &
           ((df['fare_amount'] >= 2.5) & (df['fare_amount'] < 500))]
    
    return(df)
Train = proper_data(Train)


# In[15]:


Test = Test[((Test['pickup_longitude'] > -78) & (Test['pickup_longitude'] < -70)) & 
           ((Test['dropoff_longitude'] > -78) & (Test['dropoff_longitude'] < -70)) & 
           ((Test['pickup_latitude'] > 37) & (Test['pickup_latitude'] < 45)) & 
           ((Test['dropoff_latitude'] > 37) & (Test['dropoff_latitude'] < 45)) & 
           (Test['passenger_count'] > 0) ]


# In[16]:


numerical_features = Train.columns[1:]
numerical_features


# In[17]:


import gc

gc.collect();


# In[18]:


print('Distributions columns')
plt.figure(figsize=(30, 185))
for i, col in enumerate(numerical_features):
    plt.subplot(50, 4, i + 1)
    plt.hist(Train[col]) 
    plt.title(col)
gc.collect();


# In[19]:


print('Distributions columns Test')
plt.figure(figsize=(30, 185))
for i, col in enumerate(numerical_features):
    plt.subplot(50, 4, i + 1)
    plt.hist(Test[col]) 
    plt.title(col)
gc.collect();


# In[20]:


#Scatter plots
cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
sns.scatterplot(x="passenger_count", y="fare_amount", data= Train, palette="Set2")


# In[21]:


sns.scatterplot(x="month", y="fare_amount", data= Train, palette="Set2")



# In[22]:


sns.scatterplot(x="weekday", y="fare_amount", data= Train, palette="Set2")


# In[23]:


sns.scatterplot(x="hour", y="fare_amount", data= Train, palette="Set2")


# In[24]:


#Outlier analysis
Train.plot(kind='box', subplots=True, layout=(8,3), sharex=False, sharey=False, fontsize=8)
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top= 3,wspace=0.2, hspace=0.2)
plt.show()


# In[25]:


def outliers_analysis(df): 
    for i in df.columns:
        print(i)
        q75, q25 = np.percentile(df.loc[:,i], [75 ,25])
        iqr = q75 - q25

        min = q25 - (iqr*1.5)
        max = q75 + (iqr*1.5)
        print(min)
        print(max)
    
        df = df.drop(df[df.loc[:,i] < min].index)
        df = df.drop(df[df.loc[:,i] > max].index)
        return(df)

train = outliers_analysis(Train)
test = outliers_analysis(Test)
#(15647, 11) after outlier removed


# In[26]:


## Splitting DataSets######
X_train = Train.loc[:,Train.columns != 'fare_amount']
y_train = Train['fare_amount']


# In[27]:


############################ Feature Scaling ##############################
# #Normalisation
def Normalisation(df):
    for i in df.columns:
        df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())
        
Normalisation(X_train)
Normalisation(Test)


# In[28]:


plt.hist(X_train[col]) 


# In[29]:


########################### Feature Selection ############################# 
##Correlation analysis
#Correlation plot
def Correlation(df):
    df_corr = df.loc[:,df.columns]
    sns.set()
    plt.figure(figsize=(9, 9))
    corr = df_corr.corr()
    sns.heatmap(corr, annot= True,fmt = " .3f", linewidths = 0.5,
            square=True)
    
Correlation(X_train)
Correlation(Test)


# In[30]:


######### Dimension Reduction ########
pca = PCA(n_components=10)
pca.fit(X_train)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)
plt.show()


# In[31]:


#lets reduce our no. of variables to 5 as it explains 100% features of our Data
pca = PCA(n_components=5)
X = pca.fit(X_train).transform(X_train)
Test = pca.fit(Test).transform(Test)


# In[32]:


###### Sampling the splits through stratified way ###########
X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.2)


# In[33]:


np.savetxt("Training data preprocessed.csv" , X_train, delimiter=",")


# In[34]:


###################################### MACHINE LEARNING MODELLING ###############################

####### Linear Regression #########


import statsmodels.api as sm

# Train the model using the training sets
model = sm.OLS(y_train, X_train).fit()
model.summary()


# In[35]:


predictions_LR = model.predict(X_test)


# In[36]:


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))

rmse(y_test, predictions_LR)

#17.260617483510572


# In[37]:


###### DecisionTree Modelling ##########


from sklearn.tree import DecisionTreeRegressor
fit_DT = DecisionTreeRegressor(max_depth=2).fit(X_train, y_train)


# In[38]:


predictions_DT = fit_DT.predict(X_test)


# In[39]:


rmse(y_test, predictions_DT)

#12.87861610174419


# In[40]:


rf_model = RandomForestRegressor(max_depth= 5, n_estimators = 100).fit(X_train , y_train)
rf_pred_train = rf_model.predict(X_train)
rf_pred= rf_model.predict(X_test)


# In[41]:


rmse(y_test, rf_pred)

#12.834348175666173

