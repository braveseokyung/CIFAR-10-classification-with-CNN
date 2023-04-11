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
batch_size=128
weight_decay=0.0001
momentum=0.9
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
    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
])

valid_transform=transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
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

train_loader=DataLoader(train_data,batch_size=batch_size,sampler=train_sampler,shuffle=False)
valid_loader=DataLoader(valid_data,batch_size=batch_size,sampler=valid_sampler,shuffle=False)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

# normalize by pixel mean
print(len(valid_loader.sampler))
print(len(test_loader.sampler))
pixel_mean=train_data.data.mean(axis=0)/255
pixel_mean=pixel_mean.transpose(2,0,1)
print(pixel_mean.shape)


#CBAM
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class channel_attention(nn.Module):
    
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        
        self.shared_mlp=nn.Sequential(
            Flatten(),
            nn.Linear(in_planes, in_planes//16),
            nn.ReLU(),
            nn.Linear(in_planes//16,in_planes)
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
    

# model
class ResNet(nn.Module):
    
    in_planes=16
    out_planes=[16,32,64]
    stride=[1,2,2]
    
    def __init__(self, block, n_blocks,num_classes=10):
        super().__init__()
        
        self.conv1=nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1=nn.BatchNorm2d(16)
        self.relu=nn.ReLU()
        
        self.layer1=self._make_layer(block, n_blocks[0],self.out_planes[0],stride=self.stride[0])
        self.layer2=self._make_layer(block, n_blocks[1],self.out_planes[1],stride=self.stride[1])
        self.layer3=self._make_layer(block, n_blocks[2],self.out_planes[2],stride=self.stride[2])
        
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(64*block.expansion,num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, n_block, out_planes, stride):
        
        layers=[]
        layers.append(block(self.in_planes, out_planes, stride))
        self.in_planes=block.expansion*out_planes
        for i in range(n_block-1):
            layers.append(block(out_planes, out_planes, 1))
            #self.in_planes=block.expansion*out_planes
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.avgpool(out)
        out=torch.flatten(out,1)
        out=self.fc(out)
        
        return out
        
        

class BasicBlock(nn.Module):
    
    expansion=1
    
    def __init__(self, in_planes, out_planes,stride,CBAM=True):
        super().__init__()
        
        self.conv1=nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)
        self.bn1=nn.BatchNorm2d(out_planes)
    
        self.conv2=nn.Conv2d(out_planes, out_planes, 3, 1, 1)
        self.bn2=nn.BatchNorm2d(out_planes)
        
        self.relu=nn.ReLU()
        
        self.ca=channel_attention(out_planes)
        self.sa=spatial_attention()
        
    
        self.shortcut=nn.Sequential() # identity
    
        if stride!=1:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * out_planes)
                )
            
    def forward(self,x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=torch.mul(self.ca(out),out)
        out=torch.mul(self.sa(out),out)
        
        out+=self.shortcut(x)
        out=self.relu(out)
        
        return out
        


device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device : {device}")

ResNet20=ResNet(BasicBlock,[3,3,3])
ResNet32=ResNet(BasicBlock,[5,5,5])
ResNet44=ResNet(BasicBlock,[7,7,7])
ResNet56=ResNet(BasicBlock,[9,9,9])
ResNet110=ResNet(BasicBlock,[18,18,18])
model=ResNet56.to(device)

criterion=F.cross_entropy
optimizer=torch.optim.SGD(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay,
                           momentum=momentum
                           )
#optimizer=torch.optim.Adam(params=model.parameters(),
                        #    lr=learning_rate,
                        #    weight_decay=weight_decay
                        #    )                       
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,135])



def train(model):
    
    epoch_loss=0
    n_correct=0
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data=data.to(device)
      
        #data-=torch.Tensor(pixel_mean).to(device) #subtract by pixel mean
        target=target.to(device)
        optimizer.zero_grad()
        
        output=model(data)
        batch_loss=criterion(output,target)
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
            #data-=torch.Tensor(pixel_mean).to(device) #subtract by pixel mean
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
        torch.save(model.state_dict(), '/home/NAS_mount/sk100/ResNet56_CBAM_reverse.pt')
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
model.load_state_dict(torch.load('/home/NAS_mount/sk100/ResNet56_CBAM_reverse.pt'))
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
plt.savefig("lossResNet56_CBAM_reverse.png")



