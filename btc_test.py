import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM,Dropout
from tensorflow import keras
from keras.optimizers import adam
from keras.utils import plot_model



#load Data api from yahoo 
company='BTC-USD'
start=dt.datetime(2017,1,1)
end=dt.datetime(2021,12,31)
data =web.DataReader('BTC-USD','yahoo' ,start,end)
# data.plot
# print(data)

# Prepare Data 
scaler= MinMaxScaler(feature_range=(0,1)) #Normalization
scaled_data =scaler.fit_transform(data['Close'].values.reshape(-1,1))#-1 unknown列數/ 行數為1  /.fit_transform 稀疏矩陣（英語：sparse matrix）
prediction_days=7
# print(scaled_data)
x_train=[]
y_train=[]

for x in range(prediction_days, len(scaled_data)): # start7 end  total days-7
    x_train.append(scaled_data[x-prediction_days:x,0])#0 取矩陣內的第一個/一次取prediction_days 
    y_train.append(scaled_data[x-prediction_days,0])#0 取矩陣內的第一個/會少最後prediction_days /y train 這樣訓練行嗎?

x_train,y_train=np.array(x_train),np.array(y_train)# list-array
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))   #stock prices /timesteps/indicators

#Build the model 
model= Sequential()

model.add(LSTM(units=20,return_sequences=True,input_shape=(x_train.shape[1],1))) #units:神經元數目/return_sequences?/input_shape?
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=25))
model.add(Dropout(0.2))
model.add(Dense(units=1))


opt = adam(learning_rate=0.0001)
model.compile(loss='mean_squared_error',optimizer=opt,metrics=['mae','acc']) #,metrics=['accuracy'] 訓練模型的參數設定
history=model.fit(x_train,y_train,validation_split=0.2,epochs=25,batch_size=32)#validation_split=0.2/80%訓練資料與20%驗證資料
# acc=model.evaluate(x_train,y_train,batch_size=32)

test_start=dt.datetime(2021,1,1)
test_end=dt.datetime.now()

# test_data=web.DataReader(company,'yahoo',test_start,test_end) #重test_start到test_end     
# actual_prices=test_data['Close'].values

# total_dataset=pd.concat((data['Close'],test_data['Close']),axis=0)#pd.concat 合併(訓練/test)資料，axis=0 為直向合併

# model_inputs=total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values #:?
# model_inputs = model_inputs.reshape(-1,1)
# model_inputs =scaler.transform(model_inputs)#分兩行?

# x_test=[]
# for x in range(prediction_days,len(model_inputs)):
#     x_test.append(model_inputs[x-prediction_days:x,0])

# x_test =np.array(x_test)
# x_test =np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# predicted_prices = model.predict(x_test)
# predicted_prices = scaler.inverse_transform(predicted_prices)
# model.summary()


#Plot
# plt.plot(actual_prices,color="red",label=f"Actual {company} Price")
# plt.plot(predicted_prices,color="green",label=f"Predicted {company} Price")

# plt.title(f"{company} Alpha Price")

# plt.xlabel('Days')
# plt.ylabel(f"{company} Share Price")
# plt.legend()
# plt.show()



# #loss&&acc
# plt.plot(history.history['loss'],'r',label=f"loss")
# plt.plot(history.history['val_loss'],'g',label=f"val_loss")
# plt.title("Model Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()

# plt.plot(acc,color="red", label = "accuracy")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# real_data = [model_inputs[len(model_inputs)-prediction_days:len(model_inputs+1),0]]
# real_data = np.array(real_data)
# real_data = np.reshape(real_data,(real_data.shape[0],real_data.shape[1],1))

# prediction=model.predict(real_data)
# prefiction =scaler.inverse_transform(prediction)
# print(prefiction)

# print(scaler.inverse_transform(real_data[-1]))