#从零开始实现RNN
import math 
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import numpy as np
batch_size, num_steps = 32, 35
#这里版本不一样 自己写一个文本序列的预处理
train_iter,  vocab = d2l.load_data_time_machine(batch_size, num_steps)
#one-hot 编码 : 每个token都有一个index 但是直接用这些index学习会困难
# 于是将这些token 进行编码
F.one_hot(torch.tensor([0,2]),len(vocab))

#初始化模型参数
def get_paramss(vocab_size, num_hiddens, device):
    #输入输出与vocab_size 相同 要从相同的词表中去进行转换
    num_inputs = num_outputs = vocab_size 

    def normal(shape):
        return torch.randn(size=shape, device=device)* 0.01

    #隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens,num_hiddens))
    b_h = torch.zeros(num_hiddens,device= device)
    #输出层参数
    W_hq = normal((num_hiddens,num_outputs))
    b_q = torch.zeros(num_outputs,device= device)
    #附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device= device))

def rnn(inputs, state, params):
    #inputs形状
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状 (批量大小, 词表大小vocab_size)
    for X in inputs:
        H = torch.tanh(torch.mm(X,W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H,W_hq)+b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

