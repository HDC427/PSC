import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## Define the model
class CorrScoreModel (nn.Module):
    '''
    The deep learning model used to calculate
    the corresponding score using the linear layer
    and LogSoftMax function.
    '''
    
    def __init__(self, dim_in=3, dim_out=2, use_gpu=False):
        super(CorrScoreModel, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.linear = nn.Linear(dim_in, dim_out)
        
        if use_gpu:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print('Using device :', self.device)
    
    def forward(self, feature):
        batch_size = len(feature)
        feature = torch.tensor(feature).view(batch_size, self.dim_in).to(self.device).float()
        return F.log_softmax(self.linear(feature), dim=1)
    

    
# The CPU model used to calculate.
corr_score_model = CorrScoreModel(dim_in=3, dim_out=2)

def 