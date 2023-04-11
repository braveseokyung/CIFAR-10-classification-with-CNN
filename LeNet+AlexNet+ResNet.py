# LeNet5
# accuracy score : 73.47999572753906 %
# AlexNet 
# accuracy score : 86.95999908447266 %
# ResNet
# ResNet34(self-implemented), Adam, batch size 128 : 84.47000122070312 %


import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as  plt
from mpl_toolkits.mplot3d import Axes3D

# class 10개, channel 3개
num_classes=10
in_channels=3

SEED = 10
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

batch_size=128
epoch=50
learning_rate=1e-3
valid_ratio=0.1
momentum=0.9
weight_decay=0.0001
num_blocks=[3,4,23,3] # ResNet34, ResNet50

# 모델 종류 정하기
transform_dict={
    'LeNet':transforms.ToTensor(),
    'AlexNet':transforms.Compose([
        #transforms.Resize((227,227)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),
    'ResNet':transforms.Compose([
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]) 
}

transform_validation=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# dataset 불러오기
train_data=datasets.CIFAR10(root='/home/NAS_mount/sk100/CIFAR10',train=True, download=True, transform=transform_dict['ResNet'])
test_data=datasets.CIFAR10(root='/home/NAS_mount/sk100/CIFAR10',train=False, download=True, transform=transform_validation)

#print(type(train_data.data))
#print(train_data.data)

# validation dataset 만들기
n_valid=int(len(train_data)*valid_ratio)
n_train=int(len(train_data)*(1-valid_ratio))

train_data,valid_data=random_split(train_data,[n_train,n_valid])

labels_map={
    0:'airplane', 1:'automobile', 2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'
}

print(type(train_data[0]))


# 학습용 이미지 확인
def img_show(train_data,labels_map=labels_map):
    figure=plt.figure(figsize=(10,10))
    for i in range(1,10):
        sample_idx=random.randrange(len(train_data))
        #print(sample_idx)
        img,label=train_data[sample_idx]
        figure.add_subplot(3,3,i)
        # 3*32*32 -> 32*32*3 으로 차원 변경
        plt.imshow(img.permute(1,2,0))
        plt.title(labels_map[label])

    plt.show()
    
img_show(train_data[0])


# dataloader : mini batch 만들어주는 역할
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
valid_loader=DataLoader(valid_data,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

print(type(train_data.data))
print(len(train_loader)) # 352 : iteration 숫자 전체 data 수 / batch 수
print(len(train_loader.dataset))  # 45000 : 전체 data 수

# 신경망 정의하기

# LeNet5
class Model_LeNet5(nn.Module):
    
    # layer 정의
    def __init__(self):
        
        super(Model_LeNet5,self).__init__()
        
        # convolution layer 1
        self.conv1=nn.Conv2d(
            in_channels=in_channels, 
            out_channels=6*in_channels,
            kernel_size=5,
            stride=1,
            #groups=in_channels
            #padding=1
        )  
        
        self.bn1=nn.BatchNorm2d(6*in_channels)
        
        # maxpooling layer
        self.pooling=nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        # convolution layer 2
        self.conv2=nn.Conv2d(6*in_channels,16*in_channels,5)
        
        self.bn2=nn.BatchNorm2d(16*in_channels)
        # full conenction layer 1
        self.fc1=nn.Linear(16*5*5*in_channels,120)
        
        # full connection layer 2
        self.fc2=nn.Linear(120, 84)
        
        # output layer
        self.fc3=nn.Linear(84,num_classes)
        
    
    # layer 지나는 과정 정의       
    def forward(self,input):
        X=self.conv1(input)
        X=self.bn1(X)
        X=F.relu(X)
        X=self.pooling(X)
        X=self.conv2(X)
        X=self.bn2(X)
        X=F.relu(X)
        X=self.pooling(X)
        # flatten 3차원 tensor를 1차원으로
        X=X.view(-1,16*5*5*in_channels)
        X=self.fc1(X)
        X=F.relu(X)
        X=self.fc2(X)
        X=F.relu(X)
        X=self.fc3(X)  
    
        return X
        
# AlexNet 
class Model_AlexNet(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.ConvLayer2=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
             
        )
        
        self.ConvLayer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )

        self.Classifier=nn.Sequential(
            nn.Linear(256*3*3,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)

        )        
        
    def forward(self,input):
        
        X=self.ConvLayer1(input)
        X = X.view(X.size(0), 256*3*3)
        X=self.Classifier(X)
        
        return X

class Model_GoogleNet(nn.Module):
    
    def __init__(self):
        super().__init__()
           
# ResNet
# 1) ResNet34(torchvision) : 76.93000030517578 %
# 2) ResNet34(구현), SGD , batch size 128, epoch 50: 57.16999816894531 %
# 3) ResNet34(구현), Adam , batch size 128: 84.47000122070312 %
# 4) ResNet50(구현), Adam, epoch 100: 83.54000091552734 %       
# 5) ResNet101(구현), Adam, epoch 50 : 76.86000061035156 %
class Model_ResNet(nn.Module):
    
    def __init__(self,block,num_blocks):
        super().__init__()
        
        self.in_channel=64
        
        self.conv1=nn.Sequential(
    
            nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        
        _layers=[]
        num_channels=[64,128,256,512] # for 224*224
        strides=[1,2,2,2]
        
        for i in range(4):
            _layers.append(self._make_layer(block,num_blocks[i],num_channels[i],strides[i]))
        self.block_layers=nn.Sequential(*_layers)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.mul,10)
        
    def _make_layer(self,block,num_block,num_channel,stride):
        layers=[block(self.in_channel,num_channel,stride)]
        self.in_channel=num_channel*(block.mul)
        for j in range(num_block-1):
            layer=block(self.in_channel,num_channel,1)
            layers.append(layer)
        
        return nn.Sequential(*layers)
            
        
    def forward(self,X):
        X=self.conv1(X)
        X=self.block_layers(X)
        X=self.avgpool(X)
        X=torch.flatten(X,1)
        X=self.fc(X)
        
        return X
           
class basic_block(nn.Module):
        
    mul=1
    
    def __init__(self,in_channel,out_channel,stride): # 객체 선언과 동시에 입력받는 값
        super().__init__()
        
     
        self.block_layer=nn.Sequential(
            # 나눌 때 괄호치고 int 써줘야 함
            nn.Conv2d(int(in_channel/stride),out_channel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channel),
        )
    
        self.downsample=nn.Sequential(
            nn.Conv2d(int(in_channel/stride),out_channel,kernel_size=1,stride=stride),
            nn.BatchNorm2d(out_channel)
        )
        
    def forward(self,X):
        out=self.block_layer(X)
        out+=self.downsample(X)
        out=F.relu(out) #nn.ReLU로 하면 ReLU를 반환
        return out
     
class bottleneck_block(nn.Module):
    
    mul=4
    
    def __init__(self,in_channel,out_channel,stride):
        super().__init__()
        
        self.block_layer=nn.Sequential(
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel,out_channel,3,1,1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel,out_channels=out_channel*4,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channel*4)
        )  
        
        if(stride!=1 or in_channel):
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channel,out_channel*4,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channel*4)
            ) 
        else:
        
            self.downsample=nn.Sequential(
                nn.Conv2d(out_channel*4,out_channel*4,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channel*4)
            )   
        
    def forward(self,X):
        out=self.block_layer(X)
        out+=self.downsample(X)
        out=F.relu(out)
        
        return out
             
# Early Stopping
class EarlyStopping:
    def __init__(self,model,patience=15):
        self.accuracy=0
        self.patience=0
        self.patience_limit=patience
        self.model=model
    
    def step(self,accuracy):
        if self.accuracy<accuracy:
            self.accuracy=accuracy
            self.patience=0
            torch.save(model.state_dict(), '/home/NAS_mount/sk100/model.pt')
        else:
            self.patience+=1
            
    def is_stop(self):
        return self.patience >=self.patience_limit
              
# # 만약 gpu를 사용한다면 모델이랑 데이터를 gpu 서버에 올려야 하므로
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
print(f"Device : {device}")


# 모델 선언
# LeNet=Model_LeNet5().to(device)
# AlexNet=Model_AlexNet().to(device)      
#model=torchvision.models.resnet50(num_classes=10)
#model=Model_ResNet(basic_block,num_blocks) # self-implemented ResNet34
model=Model_ResNet(bottleneck_block,num_blocks) # self-implemented ResNet50
model=model.to(device)

# loss function
# multiclass classification은 softmax-CrossEntropy 세트로 많이 쓴대
#criterion=nn.CrossEntropyLoss() # reduction?
criterion=F.cross_entropy

# optimizer : 선언된 모델의 parameter를 가져옴
optimizer=optim.Adam(params=model.parameters(),lr=learning_rate)
# scheduler : learning rate 조절
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)

#train
def train(model):
    
    model.train()
    epoch_loss=0
    n_accurate=0
    # batch idx를 구하고 싶으면 인덱스를 반환하는 enumerate 사용
    #for img, label in train_loader:
    # for문으로 iterate하면 train_loader가  64개씩 img, label 을 한번에 가져와서 학습 
    for batch_idx, (img,label) in enumerate(train_loader):
        img=img.to(device) # 64개의 3*32*32 이미지 텐서
        label=label.to(device) # 64개의 이미지에 대한 label 1*64 tensor
        output=model(img)
        pred=torch.max(output,1)[1]
        #train_loss+=criterion(output, label) # 128개의 mean을 iteration 수만큼 더함 -> 이러면 backward에 문제가 생김
        train_loss=criterion(output, label)
        epoch_loss+=train_loss.item()
        n_accurate += (pred.cpu()==label.cpu()).sum()
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    epoch_loss=epoch_loss/len(train_loader)    
    train_accuracy=(n_accurate/len(train_loader.dataset))*100    
    print(f"Train Loss : {epoch_loss}")
    return epoch_loss, train_accuracy
    
# evaluate accuracy
def evaluate(model,data_loader):
    
    model.eval()
    loss=0
    n_accurate=0
    
    with torch.no_grad():
        for img, label in data_loader:
            img=img.to(device)
            label=label.to(device)
            output=model(img) # 4(batch size)*10 tensor
            # .item() 이 없으면 tensor(loss)를 반환
            loss+=criterion(output,label,reduction='sum').item() # reduction = sum으로 해야 scale이 맞을 듯?
            pred=torch.max(F.softmax(output,dim=1),1)[1]
            n_accurate += (pred.cpu()==label.cpu()).sum()
        
            
        loss/=len(data_loader.dataset)
        accuracy=(n_accurate/len(data_loader.dataset))*100
        
    return loss,accuracy


# main, inference
# epoch를 새로 할 때 마다 다양한 무작위 가중치에서 학습이 시작됨   
early_stop=EarlyStopping(model=model) 
loss_train = []
accr_train = []
loss_valids = []
accr_valids = []

for i in range(epoch):
    print(f'Epoch : {i + 1} ==============================')
    train_loss,train_accuracy=train(model)
    #scheduler.step()
    loss_train.append(train_loss)
    accr_train.append(train_accuracy)
    # validation loss
    eval_loss,eval_accuracy=evaluate(model,valid_loader)
    scheduler.step()
    loss_valids.append(eval_loss  )
    accr_valids.append(eval_accuracy)
    print(f'Eval Loss : {eval_loss: .4f}\nEval Accuracy : {eval_accuracy} %')
    
    early_stop.step(eval_accuracy)
    #if early_stop.is_stop():
       #print(i) 
       #torch.save(AlexNet.state_dict(), 'model2.pt')
       #break

torch.save(model.state_dict(), '/home/NAS_mount/sk100/ResNet.pt')

# test data에 대한 최종 accuracy
print("===============================")
model.load_state_dict(torch.load('/home/NAS_mount/sk100/ResNet.pt'))
test_loss,test_accuracy=evaluate(model,test_loader)
print(f'Test Loss : {test_loss: .4f}\nTest Accuracy : {test_accuracy} %')
model.load_state_dict(torch.load('/home/NAS_mount/sk100/model.pt'))
test_loss,test_accuracy=evaluate(model,test_loader)
print(f'Test Loss : {test_loss: .4f}\nTest Accuracy : {test_accuracy} %')
    

# accuracy / loss plot
x=list(range(1,len(loss_train)+1))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.plot(x,loss_train,label='train loss')
ax1.plot(x,loss_valids,label='validation loss')
ax1.set_title("Loss")
ax2.plot(x,accr_train,label='train accuracy')
ax2.plot(x,accr_valids,label='validation accuracy')
ax2.set_title("Accuracy (%)")
plt.legend()
plt.savefig("loss14.png")