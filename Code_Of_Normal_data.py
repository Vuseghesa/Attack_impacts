#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1 Importing the library
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler 
#!pip install tensorflow
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time


# In[3]:


#Read data files and provide basic information
data = pd.read_csv("C:/Users/fvuse/OneDrive/Documents/IOT_temp.csv")
# Display of 5 random lines
data.sample(5)


# In[4]:


# Display of first 5 lines
data.head()


# In[5]:


# The last 5 lines
data.tail()


# In[6]:


#The form of our data is as follows:
print("Shape of our data is : ",data.shape)


# In[7]:


#The unique Values in each column are as follows:
print("Unique values in every column \n"+'-'*25)
for i in data.columns:
    print("\t"+i+" = ",len(set(data[i])))


# In[8]:


# Count the number of "X" values in the column
nombre_X = data['out/in'].value_counts()['Out']

print(nombre_X)


# In[9]:


#Information on data characteristics
data.info()


# In[10]:


# Data description
data.describe()


# In[11]:


# Deletion of data with 'id' and 'room_id/id' labels
df = data.drop(['id','room_id/id'],axis=1)
df.head()


# In[12]:


#Data analysis
#Checking for missing values
#a) Null data
data.isnull().sum()


# In[13]:


df[['outside','inside']]=pd.get_dummies(df['out/in'])


# In[14]:


print('Total Inside Observations  :',len([i for i in df['inside'] if  i == 1]))
print('Total Outside Observations :',len([i for i in df['inside'] if  i == 0]))


# In[15]:


#b) Temperature data values
print("Temperature -> \n"+"-"*30)
print("\tTotal Count    = ",df['temp'].shape[0])
print("\tMinimum Value  = ",df['temp'].min())
print("\tMaximum Value  = ",df['temp'].max())
print("\tMean Value     = ",df['temp'].mean())
print("\tStd dev Value  = ",df['temp'].std())
print("\tVariance Value = ",df['temp'].var())


# In[16]:


#Reassembling the database and displaying the new detailed database
df = df[['noted_date','temp','out/in','inside','outside']]
df.head()


# In[17]:


print(data.columns)


# In[18]:


print(df.columns)


# In[19]:


in_temperatures = df[df["out/in"] == "In"][["temp"]]
out_temperatures = df[df["out/in"] == "Out"][["temp"]]


# In[20]:


in_temperatures = np.array(in_temperatures).reshape(-1, 1)
out_temperatures = np.array(out_temperatures).reshape(-1, 1)


# In[21]:


in_scaler = MinMaxScaler()
out_scaler = MinMaxScaler()

in_temperatures_scaled = in_scaler.fit_transform(in_temperatures)
out_temperatures_scaled = out_scaler.fit_transform(out_temperatures)


# In[22]:


in_train_size = int(len(in_temperatures_scaled) * 0.8)
out_train_size = int(len(out_temperatures_scaled) * 0.8)


# In[23]:


in_test_size = len(in_temperatures_scaled) - in_train_size
out_test_size = len(out_temperatures_scaled) - out_train_size


# In[24]:


print("In Train Size:", in_train_size)
print("In Test Size:", in_test_size)
print("Out Train Size:", out_train_size)
print("Out Test Size:", out_test_size)


# In[25]:


in_train = in_temperatures_scaled[0:in_train_size, :]
in_test = in_temperatures_scaled[in_train_size:len(in_temperatures_scaled), :]


# In[26]:


out_train = out_temperatures_scaled[0:out_train_size, :]
out_test = out_temperatures_scaled[out_train_size:len(out_temperatures_scaled), :]


# In[27]:


def dataset(data, steps=1):
    data_x, data_y = [], []
    for i in range(len(data) - steps - 1):
        a = data[i:(i + steps), 0]
        b = data[i + steps, 0]
        data_x.append(a)
        data_y.append(b)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


# In[28]:


X_train_in, y_train_in = dataset(in_train)
X_test_in, y_test_in = dataset(in_test)


# In[29]:


X_train_out, y_train_out = dataset(out_train)
X_test_out, y_test_out = dataset(out_test)


# In[30]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# In[31]:


#Model
in_model = Sequential()
in_model.add(LSTM(16, input_shape=(1, 1)))
in_model.add(Dense(1))
in_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])


# In[32]:


out_model = Sequential()
out_model.add(LSTM(64, input_shape=(1, 1)))
out_model.add(Dense(1))
out_model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])


# In[33]:


from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score


# In[34]:


early = EarlyStopping(monitor="val_loss", patience=3)


# In[35]:


#in_model
in_history = in_model.fit(X_train_in, y_train_in, epochs=100, validation_split=0.2, callbacks=[early])


# In[36]:


#out_model
out_history = out_model.fit(X_train_out, y_train_out, epochs=100, validation_split=0.2, callbacks=[early])


# In[37]:


y_pred_in = in_model.predict(X_test_in)
y_pred_in = in_scaler.inverse_transform(y_pred_in)

y_test_in = y_test_in.reshape(-1, 1)
y_test_in = in_scaler.inverse_transform(y_test_in)
print("R2 Score:", r2_score(y_test_in, y_pred_in))


# In[38]:


plt.figure(figsize=(13, 4))
plt.plot(y_test_in, label="Actual")
plt.plot(y_pred_in, label="Predicted")
plt.title("Predicted and Actual inside (Normal)")
plt.legend()
plt.show()


# In[39]:


y_pred_out = out_model.predict(X_test_out)
y_pred_out = out_scaler.inverse_transform(y_pred_out)

y_test_out = y_test_out.reshape(-1, 1)
y_test_out = out_scaler.inverse_transform(y_test_out)
print("R2 Score:", r2_score(y_test_out, y_pred_out))


# In[40]:


plt.figure(figsize=(13, 4))
plt.plot(y_test_out, label="Actual")
plt.plot(y_pred_out, label="Predicted")
plt.title("Predicted and Actual outside (Normal)")
plt.legend()
plt.show()


# In[41]:


# X_test is a temporal sequence of dimensions (batch_size, timesteps, features)
# Measuring inference time for an example
start_time = time.time()
predictions = in_model.predict(np.expand_dims(X_test_in[0], axis=0))
end_time = time.time()

inference_time_single_example = end_time - start_time


# In[42]:


# Measuring average inference time
num_samples = len(X_test_in)
total_inference_time = 0

for i in range(num_samples):
    start_time = time.time()
    predictions = in_model.predict(np.expand_dims(X_test_in[i], axis=0))
    end_time = time.time()
    total_inference_time += end_time - start_time

average_inference_time = total_inference_time / num_samples


# In[43]:


# Calculate throughput (inferences per second)
throughput = num_samples / total_inference_time

print(f"Average inference time: {average_inference_time} seconds")
print(f"Throughput : {throughput} inferences per second")

