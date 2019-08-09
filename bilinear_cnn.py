# -*- coding: utf-8 -*- 
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from torch.utils.data.dataloader import default_collate
import datetime
from compact_bilinear_pooling_3d import CompactBilinearPooling3d
import sys
f_result=open('/home/pixiym/chb/BilinearCNN-pytorch/result.txt', 'w') 
sys.stdout=f_result


time_stamp = datetime.datetime.now()
print("time_stamp_start       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
# Hyper parameters
num_epochs = 1000
batch_size = 64
learning_rate = 1

#dataset
transform = T.Compose([
    T.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
])

class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img_list = [
                os.path.join(img_path, i.split()[0]) for i in lines
            ]
            self.label_list = [float(i.split()[1]) for i in lines]
        self.img_transform = img_transform
        
    

    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]
        img = Image.open(img_path).convert('RGB').resize((224, 224)) 
        if self.img_transform is not None:
            img = self.img_transform(img)
        
        return img, label

    def __len__(self):
        return len(self.label_list)

train_data = custom_dset( "/media/pixiym/CVPR/images", "/media/pixiym/CVPR/images/train_images.txt",img_transform=transform)



test_data = custom_dset( "/media/pixiym/CVPR/images", "/media/pixiym/CVPR/images/test_images.txt",img_transform=transform)


#Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,collate_fn=default_collate,drop_last=True, pin_memory=True,
                                           shuffle=True,num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,collate_fn=default_collate,pin_memory=True,
                                          shuffle=False,num_workers=4)


time_stamp = datetime.datetime.now()
print("time_stamp_data       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 
#Module
class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.vgg16featuremap = torchvision.models.vgg16(pretrained=False).features
        self.vgg16featuremap = torch.nn.Sequential(*list(self.vgg16featuremap.children())[:-1])  # Remove pool5
        self.fc = nn.Linear(512**2, 200)
       
    def forward(self, x):
        N = x.size()[0]
        out = self.vgg16featuremap(x)
        out = out.view(N, 512, 14**2)
        out = torch.bmm(out,torch.transpose(out, 1, 2)) / (14**2)
        # signed sqrt
        out = torch.mul(torch.sign(out),torch.sqrt(torch.abs(out)+1e-12))
        # L2 normalization
        out = F.normalize(out, p=2, dim=1)
        out = self.fc(out)
        assert out.size() == (N, 200)
        return out

ALLNET = NET().cuda()

# load model
ALLNET.load_state_dict(torch.load('model/ft_last.pkl'), strict=True)






#optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ALLNET.parameters(), lr=learning_rate,weight_decay=1e-8,momentum=0.9)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()        
        labels = labels.long().cuda()      
        # Forward pass
        outputs = ALLNET(images)
        time_stamp = datetime.datetime.now()
        print("time_stamp_train1       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 
        loss = criterion(outputs, labels)
        time_stamp = datetime.datetime.now()
        print("time_stamp_train2       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        time_stamp = datetime.datetime.now()
        print("time_stamp_back1       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 
        optimizer.step()
        time_stamp = datetime.datetime.now()
        print("time_stamp_back2       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S'))
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



    # Test the model
    ALLNET.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.long().cuda()
            outputs = ALLNET(images)
            time_stamp = datetime.datetime.now()
            print("time_stamp_test1       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 
            _, predicted = torch.max(outputs.data, 1)
            time_stamp = datetime.datetime.now()
            print("time_stamp_test2       " + time_stamp.strftime('%Y.%m.%d-%H:%M:%S')) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the  test images: {} %'.format(100 * correct / total))
   # Save the model 
    torch.save(ALLNET.state_dict(), 'model/ft_all.pkl')



