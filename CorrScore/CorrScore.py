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
    
    def __init__(self, dim_in=3, dim_out=2, use_gpu=True):
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
corr_score_model = corr_score_model.cuda(corr_score_model.device)
corr_score_model.load_state_dict(torch.load('CorrScore/corr_score_model_state.pth'))

def corr_score(feature):
    '''
    Function used to calculate the corresponding score using the deep learning log_soft_max model.
    * Param(s) :
        feature,    numpy.ndarray, shape (n, 3) or (3, ) The features, can be a list.
    * Return:
        score,      numpy.ndarray, shape (n, 2) or (2, ) The scores. 
    '''
    feature = np.array(feature)
    with torch.no_grad():
        X = np.reshape(feature, (-1, 3))
        log_prob_list = corr_score_model(X).cpu()
        prob_list = log_prob_list.exp().numpy()

    if feature.shape == (3,):
        prob_list = prob_list.reshape(2,)

    return prob_list

