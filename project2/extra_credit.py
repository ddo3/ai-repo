#extra_credit

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy
import pandas
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split

### VERSION 1 pythorch #####################

def pytorch_version() -> None: # not working
    dataframe = pd.read_csv('./irisdata.csv')
    two_class = dataframe[dataframe['species'] != 'setosa']

    two_class.loc[two_class['species'] == 'virginica', 'species'] = 0
    two_class.loc[two_class['species'] == 'versicolor', 'species'] = 1

    in_vec = two_class[['petal_length', 'petal_width']]
    out_vec = two_class['species']
    plt.scatter(in_vec.values[:,0], in_vec.values[:,1], c=out_vec.values)
    plt.colorbar()
    plt.show()

    num_in = 2 # size of input attributes
    num_out = 1 # size of output

    class Network(nn.Module):

        def __init__(self):
            super(Network, self).__init__()
            self.fullyconnected1 = nn.Linear(num_in,num_out)

        def forward(self, x):
            x = self.fullyconnected1(x)
            x = F.sigmoid(x)
            return x


    model = Network()
    criterion = nn.MSELoss() # loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # try tuning the learning ratio


    num_epochs = 1000 # number of training iterations
    num_examples = two_class.shape[0]
    model.train()

    for epoch in range(num_epochs):
        for idx in range(num_examples):

            # for example `idx`, convert data to tensors so that PyTorch can use it.
            attributes = torch.tensor(in_vec.iloc[idx].values, dtype=torch.float)
            label = torch.tensor(out_vec.iloc[idx], dtype=torch.float)

            # reset the optimizer's gradients
            optimizer.zero_grad()

            # send example `idx` through the model
            output = model(attributes)

            # compute gradients based on error
            loss = criterion(output, label)

            # propegate error through network
            loss.backward()

            # update weights based on propegated error
            optimizer.step()

        if(epoch % 100 == 0):
            print('Epoch: {} | Loss: {:.6f}'.format(epoch, loss.item()))

    model.eval()

    pred = torch.zeros(out_vec.shape)

    for idx in range(num_examples):
        attributes = torch.tensor(in_vec.iloc[idx].values, dtype=torch.float)
        label = torch.tensor(out_vec.iloc[idx], dtype=torch.float)
        # save the predicted value
        pred[idx] = model(attributes).round()

    v1 = sum(pred == torch.tensor(torch.FloatTensor(out_vec.values)))

    p_val = 'Correct classifications: {}/{}'.format(v1)

    print(p_val)

pytorch_version()
### VERSION 2 #####################

def keras_version() -> None:
    #load the data
    dataBase = pandas.read_csv("irisdata.csv",header=None)
    dataSet = dataBase.values
    Input = dataSet[51:,2:4].astype(float)
    Output = dataSet[51:,4]

    encodeOutput = []
    for i in Output:
        if i == 'versicolor':
            encodeOutput.append(0)
        else:
            encodeOutput.append(1)


    def plot_data():
        plt.scatter(Input[0:50,0],Input[0:50,1],c='b',label='Versicolor')
        plt.scatter(Input[50:100,0],Input[50:100,1],c='r',label='Virginica')
        plt.xlabel('Pental length')
        plt.ylabel('Pental width')
        plt.legend(loc='upper left')
        plt.show()
        #plt.savefig('inputFigure.png')

    #plot_data()

    #split the dataset into training set and validation set with train_test_split function:

    inputTrain,inputVal,outputTrain,outputVal = train_test_split(Input,encodeOutput,test_size=0.25,shuffle=True)

    #define a model
    def modelNN():
        model = Sequential()
        model.add(Dense(1,input_dim=2,activation='sigmoid'))
        model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
        return model


    model = modelNN()
    model.fit(x=inputTrain,y=outputTrain,epochs=2000, validation_data=(inputVal,outputVal))
