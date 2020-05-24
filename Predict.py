#######################################
## This is the class used to predict the coords 
## Last modified: 22:44, 31 Mar. 2020, by Benxin.
## Added LSTM model in the prediction function.

import numpy as np
from psc_funcs import SkltDrawer
import torch
from sklearn.externals import joblib
from LSTM.LSTM_fit import LSTM_Fit
from LSTM.LSTM_func import model_pred

# The model.
look_back = 10
dim_in = 2
dim_out = 2
model = LSTM_Fit(look_back=look_back, dim_in=dim_in, dim_out=dim_out).cuda()
model.load_state_dict(torch.load('LSTM/predictor_state.pth'))
scaler = joblib.load('LSTM/scaler')


class Predict:
    '''
    Class name: Predict
    
    Used to predict the success coordinate of an object.
    Using the polynomial degression, numpy.polyfit, with 
    given size of sample (sample_size) and degree (order).

    '''
    # Static variable.
    model = model
    scaler = scaler

    max_sample_size = 10
    sample_size = 7
    order = 3
    #num_skl = 1
    dimension = 2 ## The default dim is 2.
    shape = (2,)
    dimension = 2 ## The default dim is 2.

    num_pass = 0 ## Number of passed.

    samples = np.zeros((0, 2))
    t_values = np.zeros(0)

    fresh = False   ## Whether the $coord_p is up to date or not.

    @staticmethod
    def dist(r0, r1):
        '''
        The function of distance.
        Using the infinite-norme.
        '''
        r0 = np.array(r0)
        r1 = np.array(r1)

        if r0.shape != r1.shape:
            raise Exception('The shape of r0 does not match that of r1.')
        else:
            return max(abs(r0-r1))


    def __init__(self, max_sample_size=10, sample_size=7, shape=(2,), order=3):
        self.max_sample_size = max_sample_size
        self.sample_size = sample_size
        self.shape = shape
        self.order = order
        self.dimension = np.prod(np.array(shape))
        #self.num_skl = num_skl

        ## The sample is a list of vectors of dimension $self.dimension$.
        ## 'cause we need to regress in the 1st-order tensor.
        self.samples = np.zeros((0, self.dimension))
        self.t_values = np.zeros(0)
        self.num_pass = 0

        self.fresh = False
        ## Coefficients for predit the $sample_size-th coords.
        #self.t = self.sample_size**(np.array(range(self.order+1), dtype=np.float64))
        #self.t = self.t[::-1]


    def preprocess(self):
        '''
        Function used to get the optimized size of sample
        and degree.
        ! the size given by this function is guanranteed to be 
        below $self.max_sample_size
        '''
        ## This function is not finished.
        size = self.sample_size if len(self.samples)>=self.sample_size \
            else len(self.samples)
        return size, 3


    def push(self, new_coord=None):
        '''
        Function used to push a new data in the samples,
        but does not generate the prediction.
        If the new coord is invalide, we just add $self.num_pass by 1.
        '''
        ## The new coord is invalid, i.e., pass.
        if new_coord is None:
            self.num_pass = self.num_pass + 1
            
            self.fresh = True
            return False

        else:
            new_coord = np.array(new_coord)
            new_coord = new_coord.reshape((self.dimension,))

            ## Add to the sample, modify the time value, and re-init the $num_pass.
            self.samples = np.vstack([self.samples, new_coord])
            if len(self.t_values) == 0:
                t_new = 0
            else:
                t_new = self.t_values[-1] + self.num_pass + 1
            #self.t_values = np.stack([self.t_values, t_new])
            self.t_values = np.append(self.t_values, t_new)
            ## Re-init the $num_pass as 0.
            self.num_pass = 0

            ## Keep the max_size
            if len(self.samples) > self.max_sample_size:
                diff = len(self.samples) - self.max_sample_size
                self.samples = self.samples[diff:]
                self.t_values = self.t_values[diff:]
                self.t_values = self.t_values - self.t_values[0]

            self.fresh = False

            return True



    def predict(self):
        '''
        Function used to get the predicted value,
        but does not update the sample.
        '''
        if self.fresh:
            return self.value_p

        else:
            ## if the sample is too small, we return just the new value.
            if len(self.samples) <= 3:
                self.value_p = self.samples[-1]
                self.fresh = True
                return self.value_p
            
            ## If we have 10 samples, we can use the LSTM model.
            if len(self.samples) == Predict.model.look_back:
                value_p = model_pred(Predict.model, Predict.scaler, self.samples)
                self.value_p = value_p.reshape(self.shape)
                self.fresh = True

                return self.value_p

            ## Get the optimized sample_size and degree.
            size, deg = self.preprocess()

            poly = np.polyfit(x=self.t_values[-size:], \
                            y=self.samples[-size:], deg=deg)
                
            ## The value predicted:
            ## the t_n shall be $self.t_values[-1] + num_pass + 1,
            ## well, if it does not work well, we will try to not change
            ## the prediction if an invalid new coord is passed.
            value_p = np.polyval(poly, x=self.t_values[-1]+1)

            ## Reshape the predicted value as the original shape.
            self.value_p = value_p.reshape(self.shape)
            self.fresh = True

            return self.value_p

    def update(self, new_coord):
        '''
        Function used to update the samples
        and generate the prediction.   
        '''
        ## Add to the sample, or pass.
        ## @Attention: the $self.t_values is already updated.
        ## the $self.fresh is also modified by $self.push
        self.push(new_coord)

        return self.predict()

    
    def re_init(self):
        '''
        Function used to re-initialize the predictor.
        '''
        self.samples = np.zeros((0, self.dimension))
        self.t_values = np.zeros(0)
        self.num_pass = 0
        self.value_p = np.zeros(self.dimension)
