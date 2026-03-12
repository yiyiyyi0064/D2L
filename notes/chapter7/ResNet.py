#实现残差网络ResNet
import torch 
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
#定义残差块
class Residual(nn.Module):
    #先把块定义出来
    def __init__(self, in_channels, num_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, num_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
    def forward(self, X):
        #再按照网络架构按顺序处理X
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        #残差连接的X 要通过1x1卷积层 调整channels和高宽
        if self.conv3:
            X = self.conv3(X)
        Y+=X
        #这个残差块设置在ReLU之前加
        return F.relu(Y)
blk = Residual(3,3)
#batchsize channels Height Width
X = torch.rand(4,3,6,6)
Y = blk(X)
print(Y.shape)

#ResNet网络架构
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
#resnet_block ：在一个stage中 不管有多少个残差块都是最开始缩小一次HW 其他时候只变换通道
def resnet_block(in_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, num_channels, use_1x1conv=True, stride=2))
        else: 
            blk.append(Residual(num_channels, num_channels))
    return blk
b2 = nn.Sequential(*resnet_block(64,64,2, first_block=True))
b3 = nn.Sequential(*resnet_block(64,128,2))
b4 = nn.Sequential(*resnet_block(128,256,2))
b5 = nn.Sequential(*resnet_block(256,512,2))

net= nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, 10))#

#训练
def train_model():
    batch_size, lr, num_epochs = 256, 0.05, 10
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs,lr, d2l.try_gpu())
    plt.savefig('resnet_training_curve.png')

train_model()