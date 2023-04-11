# 1) DenseNet (epoch 180) : 94.2300033569336 % - from epoch 168

import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split , RandomSampler, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

# hyperparameter
epochs=180
batch_size=64
weight_decay=0.0001
momentum=0.9a
learning_rate=0.1
valid_ratio=0.1
patience=15

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

labels_map={
    0:'airplane', 1:'automobile', 2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'
}

# data augmentation 
train_transform=transforms.Compose([
    transforms.Pad(padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
])

valid_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
])

#train/valid/test split
train_data=datasets.CIFAR10(root='/home/NAS_mount/sk100/CIFAR10',train=True, download=True, transform=train_transform)
valid_data=datasets.CIFAR10(root='/home/NAS_mount/sk100/CIFAR10',train=True, download=True, transform=valid_transform)
test_data=datasets.CIFAR10(root='/home/NAS_mount/sk100/CIFAR10',train=False, download=True, transform=valid_transform)

num_data=len(train_data)
indices=list(range(num_data))
np.random.shuffle(indices)
split=5000
train_idx,valid_idx=indices[split:],indices[:split]
train_sampler=SubsetRandomSampler(train_idx)
valid_sampler=SubsetRandomSampler(valid_idx)

# n_valid=int(len(train_data)*valid_ratio)
# n_train=int(len(train_data)*(1-valid_ratio))

# train_data,_=random_split(train_data,[n_train,n_valid])
# _,valid_data=random_split(valid_data,[n_train,n_valid])

train_loader=DataLoader(train_data,batch_size=batch_size,sampler=train_sampler,shuffle=False)
valid_loader=DataLoader(valid_data,batch_size=batch_size,sampler=valid_sampler,shuffle=False)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)


#CBAM
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class channel_attention(nn.Module):
    
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.ratio=ratio
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        
        self.shared_mlp=nn.Sequential(
            Flatten(),
            nn.Linear(in_planes, in_planes//self.ratio),
            nn.ReLU(),
            nn.Linear(in_planes//self.ratio,in_planes)
        )
        
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x):
        
        max_out=self.maxpool(x)
        #print(max_out.shape)
        max_out=self.shared_mlp(max_out)
        #print(max_out.shape)
        avg_out=self.shared_mlp(self.avgpool(x))
        out=max_out+avg_out
        out=self.sigmoid(out)
        out=out.unsqueeze(2).unsqueeze(3)
        
        return out

    
class spatial_attention(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1=nn.Conv2d(2,1,kernel_size=7,padding=3)
        self.sigmoid=nn.Sigmoid()
        
    
    def forward(self,x):
        max_out=torch.max(x,dim=1,keepdim=True)[0]
        #print(max_out.shape)
        avg_out=torch.mean(x,dim=1,keepdim=True)
        #print(avg_out.shape)
        out=torch.cat([max_out,avg_out],dim=1)
        out=self.conv1(out)
        out=self.sigmoid(out)
        
        
        return out

class DenseBlock(nn.Module):
    def __init__(self,in_planes, out_planes):
        super().__init__()
        
        self.bn=nn.BatchNorm2d(in_planes)
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_planes, out_planes,3,1,1)
        
    def forward(self,x):
        x=self.bn(x)
        x=self.relu(x)
        x=self.conv(x)
        
        return x
    
class TransitionLayer(nn.Sequential):
    CBAM=False
    
    def __init__(self,in_planes,compression_theta):
        super().__init__()
        
        
        self.conv=nn.Conv2d(in_planes,int(in_planes*compression_theta),1,1)
        self.avgpool=nn.AvgPool2d(2,2)
        self.ca=channel_attention(int(in_planes*compression_theta))
        self.sa=spatial_attention()
        
    def forward(self,x):
        x=self.conv(x)
        x=self.avgpool(x)
        
        if(self.CBAM):
           x=torch.mul(self.ca(x),x)
           x=torch.mul(self.sa(x),x) 
        
        return x
        
class BottleneckLayer(nn.Sequential):
    
    CBAM=False
    def __init__(self,in_planes, growth_rate):
        super().__init__()
        
        self.bn1=nn.BatchNorm2d(in_planes)
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_planes, 4*growth_rate,1,1)
        self.bn2=nn.BatchNorm2d(4*growth_rate)
        self.conv2=nn.Conv2d(4*growth_rate,growth_rate, 3,1,1)
        
        self.ca=channel_attention(in_planes)
        self.sa=spatial_attention()
        
    def forward(self,x):
        out=self.bn1(x)
        out=self.relu(out)
        out=self.conv1(out)
        out=self.bn2(out)
        out=self.relu(out)
        out=self.conv2(out)
        
        if(self.CBAM):
           x=torch.mul(self.ca(x),x)
           x=torch.mul(self.sa(x),x) 
        
        return torch.cat([x,out],dim=1)
        
    

class DenseNet(nn.Module):
    
    in_planes=16
    out_planes=[32,16,8]
    def __init__(self, block, total_layer,growth_rate,compression_theta):
        
        
        super().__init__()
        
        self.num_layer=((total_layer-4)//3)//2 
        self.conv1=nn.Conv2d(3,16,3,1,1)
        self.dense1=self._make_dense_layer(block,growth_rate)
        self.trans1=TransitionLayer(self.in_planes,compression_theta)
        self.in_planes=int(self.in_planes*compression_theta)
        self.dense2=self._make_dense_layer(block,growth_rate)
        self.trans2=TransitionLayer(self.in_planes,compression_theta)
        self.in_planes=int(self.in_planes*compression_theta)
        self.dense3=self._make_dense_layer(block,growth_rate)
        self.globalpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(self.in_planes,10)
        
        self.ca=channel_attention(self.in_planes)
        self.sa=spatial_attention()
    
       
    def _make_dense_layer(self,block,growth_rate):
        
        layers=[]
        
        for i in range(self.num_layer):
            layers.append(block(self.in_planes,growth_rate))
            self.in_planes+=growth_rate
            
        return nn.Sequential(*layers)
        
    def forward(self,x):
        out=self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out=torch.mul(self.ca(out),out)
        out=torch.mul(self.sa(out),out) 
        out=self.globalpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        
        return out
        
        
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device : {device}")
model=DenseNet(BottleneckLayer,100,12,0.5).to(device)
        
criterion=F.cross_entropy
optimizer=torch.optim.SGD(params=model.parameters(),
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            momentum=momentum
                            )        
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,135])
        

def train(model):
    
    epoch_loss=0
    n_correct=0
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device)
        #print(data.shape)
        #print(torch.Tensor(pixel_mean).shape)
        #data-=torch.Tensor(pixel_mean).to(device)
       #print(data)
        target=target.to(device)
        optimizer.zero_grad()
        
        output=model(data)
        batch_loss=criterion(output,target) #TypeError: forward() got an unexpected keyword argument 'reduction'
        epoch_loss+=batch_loss.item()
        
        pred=torch.max(output,dim=1)[1]
        n_correct+=(pred.cpu()==target.cpu()).sum()
        
        batch_loss.backward()
        optimizer.step()
        
    epoch_loss/=len(train_loader)
    train_accuracy=(n_correct/len(train_loader.sampler))*100
    
    return epoch_loss, train_accuracy    

def evaluate(model,data_loader):
    
    loss=0
    n_correct=0
    
    model.eval()
   

    with torch.no_grad():
        for data,target in data_loader:
            data=data.to(device)
            #data-=torch.Tensor(pixel_mean).to(device)
            target=target.to(device)
            
            output=model(data)
            loss+=criterion(output,target,reduction='sum').item()
            pred=torch.max(F.softmax(output,dim=1),dim=1)[1]
            
            n_correct+=(pred.cpu()==target.cpu()).sum()
            
        #print(len(data_loader.dataset))
        loss/=len(data_loader.sampler)
        accuracy=(n_correct/len(data_loader.sampler))*100
        
    return loss, accuracy
        
    
# main
loss_train = []
accr_train = []
loss_valid = []
accr_valid = []
max_accr=0
patience_count=0
checkpoint=0

for epoch in range(epochs):
    print(f"Epoch {epoch+1} =================================")
    train_loss, train_accuracy=train(model)
    print(f"Train Loss : {train_loss}\nTrain Accuracy : {train_accuracy}%")
    loss_train.append(train_loss)
    accr_train.append(train_accuracy)

    val_loss, val_accuracy=evaluate(model,valid_loader)
    scheduler.step()
    
    loss_valid.append(val_loss)
    accr_valid.append(val_accuracy)
    print(f'Eval Loss : {val_loss: .4f}\nEval Accuracy : {val_accuracy}%')
    
    if val_accuracy>max_accr:
        torch.save(model.state_dict(), '/home/NAS_mount/sk100/DenseNet_CBAM.pt')
        max_accr=val_accuracy
        patience_count=0
        checkpoint=epoch+1
    else:
        patience_count+=1
        
    #if(patience_count>patience):
    #  break;       
        
# test data에 대한 최종 accuracy
print("===============================")

test_loss, test_accuracy=evaluate(model,test_loader)
print(f'Eval Loss : {test_loss: .4f}\nTest Accuracy : {test_accuracy}%')
# best accuracy의 state
print(checkpoint)
model.load_state_dict(torch.load('/home/NAS_mount/sk100/DenseNet_CBAM.pt'))
test_loss,test_accuracy=evaluate(model,test_loader)
print(f'Test Loss : {test_loss: .4f}\nTest Accuracy : {test_accuracy} %')

# accuracy / loss plot
x=list(range(1,len(loss_train)+1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(x,loss_train,label='train loss')
ax1.plot(x,loss_valid,label='validation loss')
ax1.set_title("Loss")
ax2.plot(x,accr_train,label='train accuracy')
ax2.plot(x,accr_valid,label='validation accuracy')  
ax2.set_title("Accuracy (%)")
plt.legend()
plt.savefig("lossDenseNet_CBAM.png")    
    