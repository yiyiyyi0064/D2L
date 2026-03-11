#MLP实现房价预测-使用Dropout控制过拟合
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import pandas as pd
import matplotlib.pyplot as plt
#加载数据
train_data=pd.read_csv('../data/kaggle_house_pred_train.csv')
test_data=pd.read_csv('../data/kaggle_house_pred_test.csv')
#数据预处理
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
#数据预处理 均值与方差标准化 均值0 方差1
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean())/x.std())
#将缺失值替换为0 这就是均值
all_features[numeric_features] = all_features[numeric_features].fillna(0)
#离散特征进行独热编码
all_features = pd.get_dummies(all_features,dummy_na=True)
#将数据转换为tensor格式
n_train = train_data.shape[0]
all_features = all_features.astype(float)
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values,dtype=torch.float32).view(-1,1)
#训练MLP
loss = nn.MSELoss()
in_features = train_features.shape[1]
def get_net():
    net =  nn.Sequential(
        nn.Linear(in_features,256),
        nn.ReLU(),
        nn.Dropout(0.5),#设置为eval模式时，pytorch自动设置dropout不工作
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128,1)
    )
    return net

def log_rmse(net,features,labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls, test_ls= [], []
    #batch_size 每份数据的大小 每次迭代使用多少数据进行训练（更新lr）
    train_iter =  d2l.load_array((train_features, train_labels), batch_size)
    #用adam优化器调整rate
    optimizer = torch.optim.Adam(net.parameters(), lr= learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        net.train()#训练模式
        for X,y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            optimizer.step()
        net.eval()#评估模式
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls

#K折交叉验证-data
def get_k_fold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0]//k
    x_train, y_train = None, None
    x_valid, y_valid = X[i*fold_size:(i+1)*fold_size,:],y[i*fold_size:(i+1)*fold_size]
    for j in range(k):
        if j== i:
            continue
        idx = slice(j*fold_size,(j+1)*fold_size)
        if x_train is None:
            x_train, y_train = X[idx,:],y[idx]
        else:#直接拼接数据 dim=0 按行拼接
            x_train = torch.cat((x_train,X[idx,:]),dim=0)
            y_train = torch.cat((y_train,y[idx]),dim=0)
    return x_train, y_train, x_valid, y_valid
#训练k次后返回平均误差
def k_fold(k, x_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls_sum,test_ls_sum= 0,0
    for i in range(k):
        data = get_k_fold_data(k,i,x_train,y_train)
        net =get_net()
        train_ls, test_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        #每次训练后将误差相加 (取的是最后一个epoch的误差)
        train_ls_sum += train_ls[-1]
        test_ls_sum += test_ls[-1]
        if i == 0: # 只绘制第一个折的结果
            #存储绘图结果到result文件夹下
            d2l.plot(list(range(1,num_epochs+1)), [train_ls, test_ls], xlabel='epoch', ylabel='rmse',
                     xlim=[1,num_epochs],ylim=[0, 0.2],legend=['train','test'])
            plt.savefig('results/train_test_rmse_mlp.png', dpi=300, bbox_inches='tight')
            
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(test_ls[-1]):f}')
    #最后返回的是k个fold的误差的平均值
    return train_ls_sum/k, test_ls_sum/k

#调用部分：参数选取
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.005,0.01, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred (train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net=get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(list(range(1,num_epochs+1)), [train_ls], xlabel='epoch', ylabel='rmse',
             xlim=[1,num_epochs],ylim=[0, 0.2],legend=['train'])
    plt.savefig('results/train_rmse_mlp.png', dpi=300, bbox_inches='tight')
    print(f'训练log rmse: {float(train_ls[-1]):f}')
    #对测试数据进行预测
    preds = net(test_features).detach().numpy()
    #将预测结果保存到kaggle_submission.csv文件中
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = test_data[['Id','SalePrice']]
    submission.to_csv('kaggle_submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)