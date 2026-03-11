import hashlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 解决 MKL 冲突
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 屏蔽 GPU 冲突
import numpy as np
import requests
import tarfile
import zipfile
import matplotlib.pyplot as plt
DATA_HUB=dict()
DATA_URL='http://d2l-data.s3-accelerate.amazonaws.com/'
#下载data
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
#解压数据
def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
#print(train_data.shape)
#print(test_data.shape)
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
#训练baseline：linear regression
loss = nn.MSELoss() 
in_features = train_features.shape[1]
def get_net():
    net = nn.Linear(in_features,1)
    return net
#关注相对误差
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
#训练函数
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls= [], []
    #batch_size 每份数据的大小 每次迭代使用多少数据进行训练（更新lr）
    train_iter =  d2l.load_array((train_features, train_labels), batch_size)
    #用adam优化器调整rate
    optimizer = torch.optim.Adam(net.parameters(), lr= learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

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
                     xlim=[1,num_epochs],ylim=[0, 0.01],legend=['train','test'])
            plt.savefig('results/train_test_rmse.png', dpi=300, bbox_inches='tight')
            
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(test_ls[-1]):f}')
    #最后返回的是k个fold的误差的平均值
    return train_ls_sum/k, test_ls_sum/k

#调用部分：参数选取
k, num_epochs, lr, weight_decay, batch_size = 10, 100, 5, 0.005, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

def train_and_pred (train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net=get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(list(range(1,num_epochs+1)), [train_ls], xlabel='epoch', ylabel='rmse',
             xlim=[1,num_epochs],ylim=[0, 0.01],legend=['train'])
    plt.savefig('results/train_rmse.png', dpi=300, bbox_inches='tight')
    print(f'训练log rmse: {float(train_ls[-1]):f}')
    #对测试数据进行预测
    preds = net(test_features).detach().numpy()
    #将预测结果保存到kaggle_submission.csv文件中
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = test_data[['Id','SalePrice']]
    submission.to_csv('kaggle_submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)