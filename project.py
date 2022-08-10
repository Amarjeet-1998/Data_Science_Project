#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import tensorflow
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense


# In[2]:


st.title("Stock Market Forcasting- Reliance")


# In[3]:


def get_ticker(name):
    company = yf.Ticker(name)
    return company


# In[4]:


company1 = get_ticker("RELIANCE.NS")


# In[5]:


tickers=["RELIANCE.NS"]


# In[6]:


Reliance =yf.download(tickers,start="2012-01-01")


# In[7]:


data1 = company1.history()
st.write("""### Reliance""")


# In[8]:


st.write(company1.info['longBusinessSummary'])


# In[9]:


st.write(Reliance)
st.header("Data understanding")


# In[10]:


st.table (Reliance.describe())
st. text_input ("Here the Maximum value of share in Open column is 2856.149902 and minimum value 78.152176. the Maximum value of share in High column is 2856.149902 and minimum value 78.894859, the Maximum value of share in Low column is 2786.100098 and minimum value 77.610634. the Maximum value of share in Close column is 2819.850098 and minimum value 66.971481.")


# In[11]:


new_Reliance=Reliance[["Close"]]
series=new_Reliance.reset_index()


# In[12]:


st.header("Visualization")
fig = px.line(x=series.Date,y=series.Close,labels={"x":"Date","y":"Closing price"})
st.write("""### Line Plot""")
st.plotly_chart(fig)


# In[13]:


st.header("Actual Vs Prediction")


# In[14]:


y=np.round((20/100)*len(series),0)
y=int(y)
x=np.round((80/100)*len(series),0)
x=int(x)


# In[15]:


train=series.head(x)
test=series.tail(y)


# In[16]:


scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(series[["Close"]].values)
scaled_train=scaler.fit_transform(train[["Close"]].values)
scaled_test=scaler.fit_transform(test[["Close"]].values)


# In[17]:


x_train=[]
y_train=[]

for i in range(20,len(scaled_train)):
  x_train.append(scaled_train[i-20:i])
  y_train.append(scaled_train[i,0])

x_train=np.array(x_train)
y_train=np.array(y_train)


# In[18]:


model=Sequential()
model.add(LSTM(units=50,activation="relu",return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.1))
model.add(LSTM(units=60,activation="relu",return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=80,activation="relu",return_sequences=True))  
model.add(Dropout(0.3))
model.add(LSTM(units=120,activation="relu"))  
model.add(Dropout(0.4))
model.add(Dense(units=1))


# In[ ]:


model.compile(optimizer="adam",loss="mse")
model.fit(x_train,y_train,epochs=100)


# In[ ]:


X=[]
Y=[]

for i in range(20,len(scaled_data)):
  X.append(scaled_data[i-20:i])
  Y.append(scaled_data[i,0])

X,Y=np.array(X),np.array(Y)


# In[ ]:


y_pred=model.predict(X)


# In[ ]:


y_df=pd.DataFrame(Y,columns=["Close"])


# In[ ]:


Actual_y=scaler.inverse_transform(y_df)


# In[ ]:


y_pred_df=pd.DataFrame(y_pred,columns=["Predicted_y"])


# In[ ]:


predicted_y=scaler.inverse_transform(y_pred_df)


# In[ ]:


y_pred_df=pd.DataFrame(predicted_y,columns=["Predictions"])


# In[ ]:


Y_df=pd.DataFrame(Actual_y,columns=["Actual"])


# In[ ]:


final_prediction_df=Y_df.join(y_pred_df)


# In[ ]:


st.write(final_prediction_df)


# In[ ]:


fig =px.line(final_prediction_df,x=final_prediction_df.index,y=final_prediction_df.columns,labels={"Price":"Numbers"})
st.plotly_chart(fig)


# In[ ]:


#get the quote
data=yf.download(tickers,start="2015-01-01")
#create new dataframe
new_df = data.filter(['Close'])
#get the last 100 days close price value and covert the data frame into array
last_20_days = new_df[-20:].values
#scale the values between 0 to 1
last_20_days_scaled = scaler.transform(last_20_days)
#create empty list
x_test = []
#append the past 100 days 
x_test.append(last_20_days_scaled)
#convert the x test data into numpy array
x_test = np.array(x_test)
#reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
#get the predicted scaled price 
pred_price = model.predict(x_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


last_20_days = new_df[-20:].values


# In[ ]:


new_predictions=new_df.tail(20).values


# In[ ]:


new_pred_df=pd.DataFrame(new_predictions,columns=["new_Predictions"])


# In[ ]:


new_arr = np.append(new_predictions, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)


# In[ ]:


new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)
new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)
new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


last_20_days = new_arr_df[-20:].values
last_20_days_scaled = scaler.transform(last_20_days)
x_test = []
x_test.append(last_20_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)
new_arr = np.append(new_arr, pred_price)


# In[ ]:


new_arr_df=pd.DataFrame(new_arr,columns=["Pred"])


# In[ ]:


df2=new_df.tail(20)


# In[ ]:


df2=pd.DataFrame(df2.values,columns=["Actual"])


# In[ ]:


future_pred=new_arr_df[20:]


# In[ ]:


st.header("Forcasting")


# In[ ]:


series_future_df=future_pred.set_index(np.arange((len(series)),(len(series)+11)))
st.write(series_future_df)

# In[ ]:


fig1=plt.figure(figsize=(7,5))
plt.plot(series.Close,label="Actual",color="red")
plt.plot(series_future_df,label="Forecasted",color="blue")
plt.legend()
st.header("Entire data and forecasting")
st.plotly_chart(fig1)


# In[ ]:


fig2=plt.figure(figsize=(7,5))
plt.plot(series[["Close"]].tail(20),label="Actual",color="red")
plt.plot(series_future_df,label="Forecasted",color="blue")
plt.legend()
st.plotly_chart(fig2)


# In[ ]:




