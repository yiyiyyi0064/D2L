#用马尔可夫假设训练MLP
import torch 
from torch import nn
from d2l import torch as d2l
import matplotlib.pylab as plt
from torch.nn import functional as F
#生成序列数据：使用正弦函数以及可加性噪声
def generate_data(T):
    time = torch.arange(1,T+1, dtype=torch.float32)
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    d2l.plot(time, [x], 'time', 'x', xlim = [1,1000], figsize=(6,3))
    plt.savefig("generate_data")
    return time,x
#使用MLP
#这里一开始实现有问题：forward中只能使用之前在init中定义过的层 因为里面有需要学习的参数
#这样才能后面去学习 
class MLP(nn.Module):
    def __init__(self, input, output):
        super().__init__() #自己初始化参数
        self.linear1 = nn.Linear(input, output)
        self.linear2 = nn.Linear(output, 1)
    def forward(self,X):
        Y = F.relu(self.linear1(X))
        Y = self.linear2(Y)
        return Y


#使用平方损失
loss = nn.MSELoss(reduction="none")

def train(net, train_iter, loss, num_epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
        f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


if __name__ == '__main__':
    T = 1000
    time, x = generate_data(T)
    tau = 4
    features = torch.zeros((T-tau,tau))
    #features 是tau个连续步组成的 labels是对应的要预测的下一个
    for i in range(tau):
        features[:, i] = x[i:T-tau + i]
    labels = x[tau:].reshape((-1,1))

    batch_size, n_train = 16, 600 #只有前n_train个样本用来训练
    train_iter = d2l.load_array((features[:n_train], labels[:n_train]),batch_size, is_train=True)
    
    net = MLP(4,10)
    train(net, train_iter, loss, 5, 0.01)
    #完成训练来进行一下预测
    onestep_preds = net(features)
    #onestep_preds data、grad等与模型参数 数据相关信息
    d2l.plot([time, time[tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000],
         figsize=(6, 3))
    plt.savefig("onestep_preds")
    #只用了前600个序列组来进行训练，
    #但是预测时我使用的是整个t-tau个序列组，但是仍然能得到较好的结果。
    #接下来后400个我要直接用预测用预测得到的feature 来进行预测-> 实现多步预测
    multistep_preds = torch.zeros(T)
    multistep_preds[: n_train+tau] = x[: n_train + tau]
    for i in range(n_train+tau, T):
        #用预测得到的继续往下预测
        multistep_preds[i] = net(multistep_preds[i-tau:i].reshape((1,-1)))
    d2l.plot([time, time[tau:], time[n_train+tau: ]],[x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
    plt.savefig("multistep_preds")


