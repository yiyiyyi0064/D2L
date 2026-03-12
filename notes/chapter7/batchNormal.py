#实现批量归一化
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
#定义batchNorm函数
def batchNorm(X, gamma, beta, moving_Mean, moving_var, eps, momentum):
    if not torch.is_grad_enabled():
        #推理过程，使用全局
        X_hat = (X - moving_Mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2: #全连接层，计算特征维上的均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else: #卷积层 N C H W 计算通道维
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        #训练中 用当前均值和方差来标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        #更新全局均值和方差
        moving_Mean = momentum * moving_Mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta #缩放和移位
    return Y, moving_Mean, moving_var
#定义BatchNorm层
class BatchNorm(nn.Module):
    # num_features：完全连接层的输出数量或卷积层的输出通道数。
    # num_dims：2表示完全连接层，4表示卷积层
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape)) #缩放参数
        self.beta = nn.Parameter(torch.zeros(shape)) #偏移参数
        self.moving_Mean = torch.zeros(shape) #全局均值
        self.moving_var = torch.ones(shape) #全局方差
    def forward(self, X):
        if self.moving_Mean.device != X.device:
            self.moving_Mean = self.moving_Mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_Mean, self.moving_var = batchNorm(
            X, self.gamma, self.beta, self.moving_Mean, self.moving_var,
            eps=1e-5, momentum=0.9
        )
        return Y

#在LeNet中使用batch Norm
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),BatchNorm(6, num_dims=4),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),BatchNorm(16, num_dims=4),nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    #flatten 就是展平 为全连接层做准备
    nn.Flatten(), nn.Linear(16 * 4 * 4, 120),BatchNorm(120, num_dims=2),nn.Sigmoid(),
    nn.Linear(120, 84),BatchNorm(84, num_dims=2),nn.Sigmoid(),
    nn.Linear(84, 10)
)
#网络结构
X = torch.rand(1, 1, 28, 28)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'Output shape: ', X.shape)
#训练模型
def train_model():
    batch_size, lr, num_epochs = 256, 0.001, 5
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.savefig('batch_norm_convergence.png')

train_model()