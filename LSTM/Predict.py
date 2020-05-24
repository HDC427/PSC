'''
This is the class used to predict the coords, using LSTM deep learning model.
Created by Benxin, 30 Mar. 2020
'''

import numpy as np
import torch
from sklearn.externals import joblib
import LSTM_fit
from LSTM_func import model_pred

look_back = 10
dim_in = 2
dim_out = 2

# The model used to predict.
model = LSTM_fit.LSTM_Fit(look_back=look_back, dim_in=dim_in, dim_out=dim_out)  
model.load_state_dict(torch.load('predictor_state.pth'))
# The scaler associated.
scaler = joblib.load('scaler')


class Predict:
    # the model used to predict:
    model = model
    scaler = scaler
    


