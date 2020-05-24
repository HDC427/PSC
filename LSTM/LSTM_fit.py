import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Fit(nn.Module):
    
    def __init__(self, look_back=10, dim_in=1, dim_out=1, dim_hid=12):
        super(LSTM_Fit, self).__init__()
        self.look_back = look_back
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        
        # LSTM
        self.lstm = nn.LSTM(dim_in, dim_hid, batch_first=True)
        self.linear = nn.Linear(dim_hid, dim_out)
        #self.init_hid()
        
    def init_hid(self, num_batch=1):
        self.hid = (torch.zeros(1,num_batch,self.dim_hid).cuda(), 
                   torch.zeros(1,num_batch,self.dim_hid).cuda())
        return self.hid
    
    def forward(self, X):
        num_batch = len(X)
        self.init_hid(num_batch)
        # Since we have setted batch_first,
        # X shall has the shape (num_batch, len_seq, dim_in)
        X = torch.tensor(X).view(num_batch, self.look_back, self.dim_in).cuda().float()
        #self.hid = self.hid.cuda()
        lstm_out, self.hid = self.lstm(X, self.hid)
        #print('lstm_out', lstm_out.shape)
        output = self.linear(lstm_out.contiguous().view(-1, self.dim_hid))
        # Reshape the linear output.
        output = output.view(num_batch, self.look_back, self.dim_out)
        # just the last output of each sequence.
        output = output[:,-1,:]
        #debug:
        # Just return the last output.
        return output.view(num_batch, self.dim_out)