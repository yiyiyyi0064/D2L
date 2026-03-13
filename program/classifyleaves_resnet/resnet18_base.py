#使用ResNet处理树叶分类问题
#这里自己简单复习一下已经学的 就不用微调了 后续做cifar的时候再用微调
import pandas as pd
import os
import torch
from torch import nn
from d2l import torch as d2l
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy  as np
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    #标准化 ImageNet的统计值
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#验证集变换
val_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#读取图片
class LeafDataset(Dataset):
    def __init__(self, df, img_dir, transform = None):
        self.df = df
        #self.img_dir = os.path.join(img_dir,"image")
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        #获取文件名和标签
        img_name = self.df.iloc[index]['image']
        label = self.df.iloc[index]['label']

        #得到完整路径
        img_path = os.path.join(self.img_dir,img_name)

        image = Image.open(img_path).convert("RGB")

        #数据增广
        if self.transform:
            image = self.transform(image)
        return image, label
    

#定义ResNet18网络架构
#先定义残差块  
class Residual(nn.Module):
    def __init__(self, in_channels, num_channels, Conv2d1x1=False, stride=1):
        super().__init__()
        

#多线程加载数据
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    #读取数据
    data_dir = r"C:\Users\yiyiyyi\.cache\kagglehub\competitions\classify-leaves"
    csv_path = os.path.join(data_dir,"train.csv")
    train_df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(
    train_df, 
    test_size= 0.2,
    random_state=42,
    stratify= train_df['label']       )

    train_date = LeafDataset(train_df, data_dir, transform=train_transform )
#这里validation data还需要从train部分分出来
    val_data = LeafDataset(val_df, data_dir, transform= val_transform)
#加载Dataloader
    train_loader = DataLoader(train_date, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    print(f"Torchvision 数据加载就绪！输入形状: {next(iter(train_loader))[0].shape}")
