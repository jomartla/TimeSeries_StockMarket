# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:13:48 2020

@author: usuario
"""


import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
#from keras.optimizers import SGD
from keras import optimizers


        
#################### CONFIGURATION ##############################

instrumentName = 'IBM'
path = './dataset'
outputImagePath = './img'
modelToRun = 'CNN1d' #'LSTM' or 'CNN1d'
N_timesteps = 20

#################################################################

dataset = pd.read_csv('{}/{}_2006-01-01_to_2018-01-01.csv'.format(path,instrumentName), index_col='Date', parse_dates=['Date'])
dataset.head()

training_set = dataset[:'2016'].iloc[:,1:2].values
test_set = dataset['2017':].iloc[:,1:2].values
training_set_volumes = dataset[:'2016'].iloc[:,4:5].values
test_set_volumes = dataset['2017':].iloc[:,4:5].values


# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
training_set_volumes_scaled = sc.fit_transform(training_set_volumes)
test_set_scaled = sc.fit_transform(test_set)
test_set_volumes_scaled = sc.fit_transform(test_set_volumes)


# Since LSTMs store long term memory state, we create a data structure with N timesteps and 1 output
# So for each element of training set, we have N previous training set elements

X_train = []
y_train = []
for i in range(N_timesteps,len(training_set)):
    register = []
    contador = -1
    for day_price in training_set_scaled[i-N_timesteps:i,0]:
        contador += 1
        register.append(day_price)
        register.append(training_set_volumes_scaled[i-N_timesteps:i,0][contador])
    X_train.append(register)
    y_value = training_set_scaled[i,0]
    if X_train[-1][-1] <= y_value:
        y_train.append(1) # One mean the price goes UP
    else:
        y_train.append(0) # One mean the price goes DOWN
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))



#################### LSTM_MODEL ##############################

if 'LSTM' in modelToRun:

    activationFunction = 'softmax'
    modelName = 'LSTM'
    
    # The LSTM architecture
    model = Sequential()
    # First LSTM layer with Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.2))
    # Second LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Third LSTM layer
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Fourth LSTM layer
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # The output layer
    model.add(Dense(units=2))
    model.add(Activation(activationFunction))
    
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # Compiling the RNN
    model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    # Fitting to the training set
    model.fit(X_train,y_train,epochs=15,batch_size=32)

#################################################################


#################### CNN1d_MODEL ################################

if 'CNN1d' in modelToRun:

    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Conv1D, GlobalAveragePooling1D
    
    kernel = 8
    activationFunction = 'sigmoid'
    modelName = 'CNN1d'
    
    model = Sequential()
    model.add(Conv1D(32, kernel_size= kernel, activation='relu', input_shape=(X_train.shape[1],1)))
    model.add(Conv1D(64, kernel_size= kernel, activation='relu'))
    #model.add(MaxPooling1D(kernel))
    model.add(Conv1D(64, kernel_size= kernel, activation='relu'))
    model.add(Conv1D(128, kernel_size= kernel, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=activationFunction))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=16, epochs=20)

#################################################################


# Preparing test data and predicting the prices
X_test = []
y_test = []
for i in range(N_timesteps,len(test_set)):
    register = []
    contador = -1
    for day_price in test_set_scaled[i-N_timesteps:i,0]:
        contador += 1
        register.append(day_price)
        register.append(test_set_volumes_scaled[i-N_timesteps:i,0][contador])
    X_test.append(register)
    y_value = test_set_scaled[i,0]
    if X_test[-1][-1] <= y_value:
        y_test.append(1) # One mean the price goes UP
    else:
        y_test.append(0) # One mean the price goes DOWN
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

predicted_stock_price_train = model.predict(X_train)
predicted_stock_price_test = model.predict(X_test)

#plot_predictions(predicted_stock_price_train,predicted_stock_price_test)


# Brief look into the prediction
up = 0
down = 0
for element in predicted_stock_price_test:
    if activationFunction == 'softmax':
        if element[0]>=element[1]:
            down += 1
            #print('down')
        else:
            up += 1
            #print('up')
    else:
        if element<=0.5:
            down += 1
            #print('down')
        else:
            up += 1
            #print('up')


# Real accuracy calculation on testing set
correct = 0
incorrect = 0
contador = -1
for element in predicted_stock_price_train:
    contador += 1
    if activationFunction == 'softmax':
        if element[0]>=element[1]:
            direction = 0
        else:
            direction = 1
        if direction == y_train[contador]:
            correct += 1
        else:
            incorrect += 1
    else:
        if element<=0.5:
            direction = 0
        else:
            direction = 1
        if direction == y_train[contador]:
            correct += 1
        else:
            incorrect += 1
accuracy = correct/(correct+incorrect)
print('\nAccuracy on training set:\n' + str(accuracy))


# Real accuracy calculation on testing set
correct = 0
incorrect = 0
contador = -1
for element in predicted_stock_price_test:
    contador += 1
    if activationFunction == 'softmax':
        if element[0]>=element[1]:
            direction = 0
        else:
            direction = 1
        if direction == y_test[contador]:
            correct += 1
        else:
            incorrect += 1
    else:
        if element<=0.5:
            direction = 0
        else:
            direction = 1
        if direction == y_test[contador]:
            correct += 1
        else:
            incorrect += 1
accuracy = correct/(correct+incorrect)
print('\nAccuracy on testing set:\n' + str(accuracy))


# Plotting the information
datasetForPlot = dataset['2017':]
datasetForPlot2 = dataset["High"]['2017':]
valueForIndicator = min(datasetForPlot2) * 0.95
upPredictions = []
downPredictions = []
for i in range(0,len(datasetForPlot2)):
    if i < 60:
        upPredictions.append(None)
        downPredictions.append(None)
    else:
        if activationFunction == 'softmax':
            if predicted_stock_price_test[i-60][0]>=predicted_stock_price_test[i-60][1]:
                prediction = 0
            else:
                prediction = 1        
            if prediction == 1:
                upPredictions.append(valueForIndicator)
                downPredictions.append(None)
            else:
                upPredictions.append(None)
                downPredictions.append(valueForIndicator)
        else:
            if predicted_stock_price_test[i-60]<=0.5:
                prediction = 0
            else:
                prediction = 1        
            if prediction == 1:
                upPredictions.append(valueForIndicator)
                downPredictions.append(None)
            else:
                upPredictions.append(None)
                downPredictions.append(valueForIndicator)
    
datasetForPlot['upPredictions'] = upPredictions
datasetForPlot['downPredictions'] = downPredictions


datasetForPlot2.plot(figsize=(16,4),legend=True)
if pd.isnull(datasetForPlot['upPredictions']).all() != True:
    datasetForPlot['upPredictions'].plot(figsize=(16,8),legend=True,color='green')
if pd.isnull(datasetForPlot['downPredictions']).all() != True:
    datasetForPlot['downPredictions'].plot(figsize=(16,8),legend=True,color='red')
plt.legend(['Test set (2017 and beyond)','Prediction of the price going up','Prediction of the price going down'])
plt.title('{} stock price'.format(instrumentName))
plt.savefig(outputImagePath + '/' + instrumentName + '_' + modelName + '_{}' + N_timesteps + '.png')
plt.show()
