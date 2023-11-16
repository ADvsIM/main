import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class FocalLoss(nn.Module):

    r"""
    Paper: Focal Loss for Dense Object Detection
    Summary: 잘 분류되지 않은 sample의 가중치와 minor class에 대한 가중치 도입한 loss
    function: FL(pt) = -alpha_t*((1-pt)^gamma)*log(pt)

    alpha: minor class의 weight, ex) label의 0:1 비율이 1000:1 => alpha = [0.001,1]
    gamma: 잘 분류되지 않은 sample의 weight. 지수 형태이므로 가중치를 주고싶다면 1이상으로 지정. 
    logpt: log softmax prediction value의 true label에 있는 값. [0.2, 0.8] 인데 라벨이 1이라면 0.8로 지정.

    """

    def __init__(self, device, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.size_average = size_average
        self.device = device

        # make a alpha list
        self.alpha = alpha
        if isinstance(alpha,(float,int)): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target, activation = None):
        
        input = input.reshape(-1,1).to(self.device)
        target = target.reshape(-1,1).to(self.device)
        self.alpha = self.alpha.to(self.device)

        if activation == 'softmax':
            logpt = F.log_softmax(input)
        elif activation == 'sigmoid': 
            pt = nn.Sigmoid()(input)
            logpt = torch.cat((torch.log(1-pt), torch.log(pt)), dim = 1).to(self.device)
        else:
            logpt = torch.cat((torch.log(1-input), torch.log(input)), dim = 1).to(self.device)
        
        logpt = logpt.gather(1,target.to(torch.int64).view(-1, 1))
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            # if self.type()!=input.type():
            #     self.alpha = self.alpha.type_as(input)
            at = self.alpha.gather(0,target.to(torch.int64))
            alpha_logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * alpha_logpt

        if self.size_average: 
            return loss.mean()
        else: 
            return loss.sum()
        


def fpe_fne(input, target):

    num_negative = (target == 0).sum().item()
    num_positive = (target == 1).sum().item()


    if num_negative == 0 :
        fpe = 0
    else:
        fp = ((input[target == 0])**2).sum()
        fpe = fp /num_negative

    if num_positive == 0:
        fne = torch.tensor(0, dtype=torch.float32)
    else:
        fn = ((input[target == 1] -1 )**2).sum()
        fne = fn /num_positive

    
    return fpe , fne


class MFELoss(nn.Module):

    r"""
    Paper: Training Deep Neural Networks on Imbalanced Data Sets 
    Summary: minor class weight를 주기 위해 False positive error와 False negative error를 동시에 줄이는 방법
    function: L(d,y) = FPE + FNE

    FPE: true 음성(y==0)인데 양성(d==1)으로 잘못 분류
    FNE: true 양성(y==1)인데 음성(d==0)으로 잘못 분류

    """

    def __init__(self, device):
        super(MFELoss, self).__init__()
        self.device = device

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)

        fpe, fne = fpe_fne(input, target)
        loss = fpe + fne

        return loss
    
        
class MSFELoss(nn.Module):

    r"""
    Paper: Training Deep Neural Networks on Imbalanced Data Sets 
    Summary: MFE loss로 구하면 negative class가 훨씬 많아서 FPE의 영향이 크므로 이를 보정해주는 방법.
    function: L(d,y) = ((FPE + FNE)^2 + (FPE - FNE)^2)/2

    FPE: true 음성(y==0)인데 양성(d==1)으로 잘못 분류
    FNE: true 양성(y==1)인데 음성(d==0)으로 잘못 분류

    """

    def __init__(self, device):
        super(MSFELoss, self).__init__()
        self.device = device

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)

        fpe, fne = fpe_fne(input, target)
        loss = fpe**2 + fne**2

        return loss
    
        

class ClassBalancedLoss(nn.Module):

    r"""
    Paper:  Class-Balanced Loss Based on Effective Number of Sample
    Summary: (# of label)에 반비례하는 loss
    function: CB(p,y) = ((1-beta)/(1-beta^n_y)) * L(p,y) 

    beta: [0,1) 범위 내의 hyperparameter, beta 값이 클수록 가중치 차이가 많이 남.
    L(p,y): 논문에서 사용한 loss로는 cross entropy, focal loss가 있었음. squared loss까지 추가.
    num_of_label_list: class별 label 개수 list. ex) [23, 1000]
    """


    def __init__(self, device,num_of_label_list ,beta=0.5, loss_type = 'cross-entropy', gamma = 0):
        super(ClassBalancedLoss, self).__init__()

        self.num_of_label_list = num_of_label_list
        self.no_of_classes = len(num_of_label_list)
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        self.device = device

    def forward(self, input, target):

        input = input.squeeze().to(self.device)
        target = target.clone().to(self.device)
        # print(input.shape, target.shape)

        effective_num = 1.0 - np.power(self.beta, self.num_of_label_list)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights)

        labels_one_hot = F.one_hot(target.to(torch.int64), self.no_of_classes).float()

        weights = torch.tensor(weights).to(self.device).float().unsqueeze(0)                     # dim : num_class > 1*num_class
        weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot     # dim : 1*num_class > batch * num_class
        weights = weights.sum(1).unsqueeze(1)                                    # dim : batch * num_class > batch > batch * 1
        # weights = weights.repeat(1,self.no_of_classes)                                # dim : batch * 1 > batch * num_class
        
        if self.loss_type == "focal":
            cb_loss = FocalLoss(self.device, gamma = self.gamma, alpha = weights)(input, target)
        elif self.loss_type == "cross-entropy":
            cb_loss = F.binary_cross_entropy(input = input ,target = target, weights = weights)
        elif self.loss_type == "squared":
            r_weights = torch.sqrt(weights)
            cb_loss = nn.MSELoss()(input = r_weights*input, target = r_weights*target)
        # elif self.loss_type == "mfe":
            
        return cb_loss
