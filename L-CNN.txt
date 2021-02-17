from osgeo import gdal
from torch.utils.data import Dataset
import os
from torch.utils import data
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms,utils
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


#构建自己的数据集
class DataProcessingMnist(Dataset):
    def __init__(self, root_path, imgfile_path, labelfile_path, imgdata_path, transform = None):
        self.root_path = root_path
        self.transform = transform
        self.imagedata_path = imgdata_path
        img_file = open((root_path + imgfile_path),'r')
        self.image_name = [x.strip() for x in img_file]
        img_file.close()
        label_file = open((root_path + labelfile_path), 'r')
        label = [int(x.strip()) for x in label_file]
        label_file.close()
        self.label = torch.LongTensor(label)
        
    def __getitem__(self, idx):
            image = gdal.Open(str(self.image_name[idx]))
            im_width = image.RasterXSize    
            im_height = image.RasterYSize   
            im_geotrans = image.GetGeoTransform()  
            image = image.ReadAsArray(0,0,im_width,im_height) 
            
            if self.transform is not None:
                    image = self.transform(image)
            label = self.label[idx]
            return image, label
        

    def __len__(self):
        return len(self.image_name)

root_path = 'D:/doctor/cnn1/'
train_path = 'train/'
test_path = 'test/'

train_imgfile =  train_path + 'train.txt'
train_labelfile = train_path + 'train_label.txt'
train_imgdata =  train_path + 'img/'

test_imgfile =  test_path + 'test.txt'
test_labelfile = test_path + 'test_label.txt'
test_imgdata =  test_path + 'img/'


trainset = DataProcessingMnist(root_path, train_imgfile, train_labelfile, train_imgdata)
trainloader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)
trainsize = len(trainset)
trainclasses = 2
print(trainsize)


testset = DataProcessingMnist(root_path, test_imgfile, test_labelfile, test_imgdata)
testloader = DataLoader(dataset=testset, batch_size=1)
testsize = len(testset)
print(testsize)


####network
class cnn(nn.Module):
    def __init__(self,num_classes=2):
        super(cnn,self).__init__()
        self.features=nn.Sequential(

            nn.Conv2d(9,32,kernel_size=3,stride=1,padding=1),   
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2,stride=2),   

            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1), 
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1), 
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),    
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=2),   
            
        )   


        self.classifier=nn.Sequential(
                        
            nn.Linear(512*2*2,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
                        
            nn.Linear(1024,num_classes),
            
        )
        
    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

net=cnn()

criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

#train
print ("training begin")
losses = []
acces = []
test_losses = []
test_acces = []

for epoch in range(80):
    start = time.time()

    train_loss=0
    train_acc = 0
    test_loss=0
    test_acc = 0
    
    for i,data in enumerate(trainloader,0):

        image,label=data

        image=image.float()
        label=label
        image=Variable(image)
        label=Variable(label)

        optimizer.zero_grad()
        output=net(image)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()


        # 记录误差
        train_loss += loss.item()

        # 计算分类的准确率
        _, pred = output.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / image.shape[0]
        train_acc += acc

    end=time.time()
    losses.append(train_loss / len(trainloader))
    acces.append(train_acc / len(trainloader))

    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}  time: {:.6f} s'.format(epoch+1, train_loss / len(trainloader), train_acc / len(trainloader),(end-start)))
    train_loss=0
    
    #test
    net.eval()
    correct=0
    for data in testloader:
        images,labels=data
        images=images.float()
        labels=labels
        outputs=net(Variable(images))

        testloss=criterion(outputs,labels)
        test_loss += testloss.item()
        
        _,predicted=torch.max(outputs,1)
        num_corrects = (predicted == labels).sum().item()
        accs = num_corrects / images.shape[0]
        test_acc += accs

    end=time.time()
    test_losses.append(test_loss / len(testloader))
    test_acces.append(test_acc / len(testloader))


    print('epoch: {}, Test Loss: {:.6f}, Test Acc: {:.6f}  time: {:.6f} s'.format(epoch+1, test_loss / len(testloader), test_acc / len(testloader),(end-start)))
    test_loss=0
   
print ("finish training")


#保存整个模型；
torch.save(net, r'D:\doctor\SCI3\cnn1\cnnnet.pth')

#在同一幅图片上画两条折线
train_accuracy,=plt.plot(np.arange(len(acces))+1,acces,'-r',label='train_accuracy',linewidth=1.0)
test_accuracy,=plt.plot(np.arange(len(test_acces))+1,test_acces,'-b',label='test_accuracy',linewidth=1.0)
train_loss,=plt.plot(np.arange(len(losses))+1,losses,'-.r',label='train_loss',linewidth=1.0)
test_loss,=plt.plot(np.arange(len(test_losses))+1,test_losses,'b-.',label='test_loss',linewidth=1.0)

#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
legend = plt.legend(handles=[train_accuracy,test_accuracy,train_loss,test_loss],prop=font1)


#设置横纵坐标的名称以及对应字体格式
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}
plt.xlabel('epoch',font2)
plt.ylabel('accuracy/loss53A',font2)

plt.show()


