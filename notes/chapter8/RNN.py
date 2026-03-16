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
def get_params(vocab_size, num_hiddens, device):
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

class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens, device, 
                    get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        #要返回参数
        return self.forward_fn(X, state, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
    
def predict(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix]]
    get_input = lambda: torch.tensor([outputs[-1]],device=device).reshape((1,1))
    #预热 即将Hidden隐状态H调整为上下文值
    for y in prefix[1:]:
        #更新H 每次预测结果直接丢弃
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    #进行预测
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        #y.argmax(dim=1) 模型输出为概率分布 max取概率最大那个字符
        #预测出来的y存入outputs 说明下一次循环调用get_input时，会把自己说的话当成输出
        outputs.append(int(y.argmax(dim=1)).reshape(1))
    return ''.join([vocab.idx_to_token[i] for i in outputs])



if __name__ == '__main__':
    num_hiddens = 512
    X = torch.arange(10).reshape((2, 5))
    F.one_hot(X.T, 28).shape
    #检查输出形状是否正确
    net = RNNModelScratch(len(28), num_hiddens, d2l.try_gpu(),
                          get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], d2l.try_gpu())
    Y, new_state = net(X.to(d2l.try_gpu()), state)
    Y.shape, len(new_state), new_state[0].shape


