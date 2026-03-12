#NiN
#定义block
import torch
import torch.nn as nn
from d2l import torch as d2l
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),nn.ReLU()
        #这里1*1卷积层 和全连接层没什么区别 增加了非线性激活函数 可以提升模型的表达能力
    )

#NiN网络结构
def nin():
    return nn.Sequential(
        nin_block(1,96,kernel_size=11,stride=4, padding=0),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96,256,kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256,384,kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(384, 10, kernel_size=3, stride=1, padding=1),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten()
    )
#查看每层输出的形状
net = nin()
X = torch.rand(size=(1,1,224,224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)