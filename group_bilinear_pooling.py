# -*- coding: utf-8 -*- 

import torch
from torch import nn
from torch.autograd import Variable
from compact_bilinear_pooling_3d  import CompactBilinearPooling3d
import os
import torch.nn.functional as F



#第一层双线性池化
CBPlayer1 = CompactBilinearPooling3d(3, 3, 4,sum_pool = False)
CBPlayer1.cuda()
CBPlayer1_vector = CompactBilinearPooling3d(3, 3, 4)
CBPlayer1_vector.cuda()
#第二层双线性池化
CBPlayer2 = CompactBilinearPooling3d(4, 4, 8,sum_pool= False)
CBPlayer2.cuda()
CBPlayer2_vector = CompactBilinearPooling3d(4, 4, 8)
CBPlayer2_vector.cuda()
#第三层双线性池化
CBPlayer3 = CompactBilinearPooling3d(8, 8, 16,sum_pool= False)
CBPlayer3.cuda()
CBPlayer3_vector = CompactBilinearPooling3d(8, 8, 16)
CBPlayer3_vector.cuda()

#第四层双线性池化
CBPlayer4_vector = CompactBilinearPooling3d(16, 16, 32)
CBPlayer4_vector.cuda()

class GroupBilinearPoolingNet(nn.Module):
    def __init__(self):
        super(GroupBilinearPoolingNet, self).__init__()
        
     
    def gbplayer(self,x):
        a  = []
        b1 = []
        b1_vector = []
        b2 = []
        b2_vector = []
        b3 = []
        b3_vector = []
        b4 = []
        for i1 in range(1534):     
          a.append(x[:,i1:i1+3,:,:])
        for j1 in range(0,512,2):   #第1次双线性池化，输出结果为256个


          out1 = CBPlayer1(a[j1], a[j1+1])
          # signed sqrt
          x = torch.mul(torch.sign(out1),torch.sqrt(torch.abs(out1)+1e-12)) 
          # L2 normalization
          out1 = F.normalize(x, p=2, dim=1)
          b1.append(out1)
          out1_vector = CBPlayer1_vector(a[j1], a[j1+1])
          # signed sqrt
          x = torch.mul(torch.sign(out1_vector),torch.sqrt(torch.abs(out1_vector)+1e-12)) 
          # L2 normalization
          out1_vector = F.normalize(x, p=2, dim=1)
          b1_vector.append(out1_vector)
        for j2 in range(0,256,2):   # 第二次双线性池化，输出结果为128个
          out2 = CBPlayer2(b1[j2], b1[j2+1])
          # signed sqrt
          x = torch.mul(torch.sign(out2),torch.sqrt(torch.abs(out2)+1e-12)) 
          # L2 normalization
          out2 = F.normalize(x, p=2, dim=1)
          b2.append(out2)
          out2_vector = CBPlayer2_vector(b1[j2], b1[j2+1])
          # signed sqrt
          out2_vector = torch.mul(torch.sign(out2_vector),torch.sqrt(torch.abs(out2_vector)+1e-12)) 
          # L2 normalization
          out2_vector = F.normalize(x, p=2, dim=1)
          b2_vector.append(out2_vector)
        for j3 in range(0,128,2):  # 第三次双线性池化，输出结果为64个
          out3 = CBPlayer3(b2[j3],b2[j3+1]) 
          # signed sqrt
          x = torch.mul(torch.sign(out3),torch.sqrt(torch.abs(out3)+1e-12)) 
          # L2 normalization
          out3 = F.normalize(x, p=2, dim=1)
          b3.append(out3)  
          out3_vector = CBPlayer3_vector(b2[j3],b2[j3+1]) 
          # signed sqrt
          out3_vector = torch.mul(torch.sign(out3_vector),torch.sqrt(torch.abs(out3_vector)+1e-12)) 
          # L2 normalization
          out3_vector = F.normalize(out3_vector, p=2, dim=1)
          b3_vector.append(out3_vector)
        for j4 in range(0,64,2):   # 第四次双线性池化，输出结果为32个
          out4_vector = CBPlayer4_vector(b3[j4],b3[j4+1]) #  此时sum_pool设为true，前三次都设为false
          # print(out4.shape)              # 32*(10,32)
          # signed sqrt
          out4_vector = torch.mul(torch.sign(out4_vector),torch.sqrt(torch.abs(out4_vector)+1e-12)) 
          # L2 normalization
          out4_vector = F.normalize(out4_vector, p=2, dim=1)
          b4.append(out4_vector)
        
        
        f = torch.cat((b1_vector, b2_vector,b3_vector,b4), 1)
       # print(f.size())              #(10,1024)
        return f 
    def forward(self, x):
        out = self.gbplayer(x)
        return out
