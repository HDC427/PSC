import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.cuda.is_available())

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from bokeh.plotting import figure, show
from bokeh.models import Title
from bokeh.io import output_notebook
from bokeh.layouts import row

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from .LSTM_fit import LSTM_Fit


def gen_dataset(data_set, look_back=10, dim_out=1):
    '''
    Function used to generate the dataset for training.
    '''
    data_X = []
    data_Y = []
    
    for i in range(len(data_set)-look_back-1):
        data_X.append(data_set[i:i+look_back, :-dim_out])
        data_Y.append(data_set[i+look_back, -dim_out:])
    #print('X : %s, Y : %s' % (data_X[-1], data_Y[-1])) # Show the form of (x,y).
    return np.array(data_X), np.array(data_Y)


def cut_dataset(data_set, look_back=10):
    if len(data_set) < look_back:
        raise('Err, len(data_set < look_back)')
        
    data_X = [data_set[i:i+look_back] for i in range(len(data_set)-look_back+1)]
    return data_X


def LSTM_train(dataset, dim_in=1, dim_out=1, look_back=10, batch_size=1280, num_epoch=50, bokeh=False):
    
    # Scaler
    scaler = MinMaxScaler()
    data_XY = scaler.fit_transform(dataset)
    
    # Split the datas
    data_X, data_Y = gen_dataset(data_XY, look_back, dim_out)
    num_data = len(data_X)
    num_train = int(2*num_data/3)
    
    # Define the model
    model = LSTM_Fit(look_back, dim_in, dim_out)
    model.cuda() # To GPU
    func_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # Now do the trainning.
    loss_list = []
    for epoch in tqdm(range(num_epoch)):
        loss_total = 0
        num_it = 0
        for i in tqdm(range(0, num_train, batch_size)):
            model.zero_grad()
            #model.init_hid()
            
            # get the batched datas.
            # The upper bound.
            j = i+batch_size if i+batch_size<num_train else num_train
            X = data_X[i:j]
            Y = torch.tensor(data_Y[i:j]).cuda().float().view(j-i,dim_out)
            
            Y_p = model(X)
            loss = func_loss(Y_p, Y)
            loss.backward()
            optimizer.step()
            loss_total += loss
            num_it += 1
            
            
        loss_mean = loss_total/num_it
        loss_list.append(loss_mean)
        print('Epoch : %4d/%4d, loss : %.3f' % (epoch, num_epoch, loss_mean))
        
    
    # Then calculate the predcited values.
    with torch.no_grad():
        data_Yp = model(data_X).cpu().numpy()
    data_XYp = np.column_stack([data_XY[look_back+1:, :-dim_out], data_Yp])
    data_pred = scaler.inverse_transform(data_XYp)
    
    # Seperate the train and test set.
    tar_pred_train = data_pred[:num_train, -dim_out:]
    tar_pred_test  = data_pred[num_train:, -dim_out:]
    tar_real_train = dataset[look_back+1:look_back+1+num_train, -dim_out:]
    tar_real_test  = dataset[look_back+1+num_train:, -dim_out:]
    
    # Calculate the train and test err.
    err_train = np.sqrt(mean_squared_error(tar_real_train, tar_pred_train))
    err_test  = np.sqrt(mean_squared_error(tar_real_test,  tar_pred_test ))
    print('Train Err : %.2f' % err_train)
    print('Test  Err : %.2f' % err_test )
    
    # Then plot the results.
    #feature,  = plt.plot(dataset[:,0], color='red', label='feature')
    for i in range(dim_out):
        target,   = plt.plot(dataset[:,-i], color='blue', label='target')
        target_p, = plt.plot(np.arange(look_back+1, len(dataset)), data_pred[:,-i],
                color='green', label='tar_pred')
        bound,    = plt.plot([num_train,num_train], [np.min(dataset[:,-i]), np.max(dataset[:,-i])], color='violet')
        plt.legend(handles=[target, target_p])
        plt.show()

    plt.plot(loss_list)
    plt.legend(['loss'])
    
    if bokeh and dim_out == 2:
        fig = figure()
        fig.line(x=dataset[look_back+1:,-2], y=dataset[look_back+1:,-1], color='blue', alpha=0.8, legend='real val')
        fig.line(x=data_pred[:num_train,-2], y=data_pred[:num_train,-1], color='orange', alpha=0.5, legend='train')
        fig.line(x=data_pred[num_train:,-2], y=data_pred[num_train:,-1], color='violet', alpha=0.5, legend='test')
        show(fig)
    
    return model, scaler
    
    
def model_pred(model, scaler, dataA):
    dim_in = dataA.shape[1]
    # Check dim in:
    if dim_in != model.dim_in:
        raise('Err, the dimension of given data does not match with input dim of the model.')
    # find the required num_col of dataset.
    dim_out = scaler.min_.shape[0]-dim_in
    if dim_out != model.dim_out:
        raise('Err, the dimension of scaler does not match with input dim of the model.')
        
    look_back = model.look_back
    # pad the data.
    dataset = np.column_stack([dataA, np.zeros((len(dataA),dim_out))])
    # scale the data
    dataXY = scaler.transform(dataset)
    # seperate data X and Y
    dataX = cut_dataset(dataXY[:,:dim_in], look_back)
    # precess the model
    with torch.no_grad():
        dataY_p = model(dataX).cpu().numpy()

    # combine the predicted data.
    dataXY_p = np.column_stack([dataXY[look_back-1:, :dim_in], dataY_p])
    # Reverse
    data_pred = scaler.inverse_transform(dataXY_p)
    # Seperate the predicted data.
    tar_pred = data_pred[:, -dim_out:]
    
    return tar_pred