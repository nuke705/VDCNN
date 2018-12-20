#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 00:05:34 2018

@author: jingw
"""


# coding: utf-8

# In[1]:


# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import os
import pandas as pd
from os.path import exists
import time

# In[2]:


def pad_text(text, pad, fixed_length):
    length = len(text)
    if length < fixed_length:
        return text + [pad]*(fixed_length - length)
    else:
        return text[:fixed_length]

class TextDataset(Dataset):
    
    def __init__(self, texts, dictionary, sort=False, fixed_length=1024):

        PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
        
        self.texts = [([dictionary.indexer(i) for i in list(str(text).lower())], label) for label, text in texts.values.tolist()]

        if fixed_length:
            self.texts = [(pad_text(text, PAD_IDX, fixed_length), label) for text, label in self.texts]
            
        if sort:
            self.texts = sorted(self.texts, key=lambda x: len(x[0]))
        
    def __getitem__(self, index):
        tokens, label = self.texts[index]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
    def __len__(self):
        return len(self.texts)

class VDCNNDictionary:

    def __init__(self, args=None):

        self.ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%^&*~`+=<>()[]{}" # + space, pad, unknown token
        self.PAD_TOKEN = '<PAD>'
        
        self.build_dictionary()

    def build_dictionary(self):

        self.vocab_chars = [self.PAD_TOKEN, '<UNK>', '<SPACE>'] + list(self.ALPHABET)
        self.char2idx = {char:idx for idx, char in enumerate(self.vocab_chars)}
        self.vocabulary_size = len(self.vocab_chars)

    def indexer(self, char):
        if char.strip() == '':
            char = '<SPACE>'
        try:
            return self.char2idx[char]
        except:
            char = '<UNK>'
            return self.char2idx[char]
            
# glove_embeddings = np.load('glove_embeddings.npy')

# class TextDataLoader(DataLoader):
    
#     def __init__(self, dictionary, *args, **kwargs):
#         super(TextDataLoader, self).__init__(*args, **kwargs)
#         self.collate_fn = self._collate_fn
#         self.PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
    
#     def _collate_fn(self, batch):
#         text_lengths = [len(text) for text, label in batch]
        
#         longest_length = max(text_lengths)

#         texts_padded = [pad_text(text, pad=self.PAD_IDX, min_length=longest_length) for text, label in batch]
#         labels = [label for text, label in batch]

#         #print(np.array(texts_padded).shape)
#         texts_padded = glove_embeddings[texts_padded]
#         #print(np.array(texts_padded).shape)        

#         texts_tensor, labels_tensor = torch.FloatTensor(texts_padded), torch.LongTensor(labels)
#         return texts_tensor, labels_tensor


# In[3]:


#yelp full review
# ypf1 = './datasets/yelp_review_full_csv/train.csv'
# ypf2 = './datasets/yelp_review_full_csv/test.csv'

#ypf1 = 'E:/study/CS598/project/yelp_review_full_csv/train.csv'
#ypf2 = 'E:/study/CS598/project/yelp_review_full_csv/test.csv'
#ag news
#ag1 = './datasets/ag_news_csv/train.csv'
#ag2 = './datasets/ag_news_csv/test.csv'
#yahoo news
yh1 = './datasets/yahoo_answers_csv/train.csv'
yh2 = './datasets/yahoo_answers_csv/test.csv'
train = pd.read_csv(yh1, header=None, usecols = [0,1,2,3],parse_dates = [[1,2,3]])
#train = pd.read_csv('E:/study/CS598/project/yelp_review_full_csv/train.csv', header=None)
train.columns = ['star', 'review']
tempstar = train['star']
train['star']=train['review']
train['review']= tempstar
train['star'] =  train['star']  - 1
#train.columns = ['star', 'title','review']

#test = pd.read_csv('E:/study/CS598/project/yelp_review_full_csv/test.csv', header=None)
test = pd.read_csv(yh2, header=None, usecols = [0,1,2,3],parse_dates = [[1,2,3]])
test.columns = ['star', 'review']
tempstart = test['star']
test['star']=test['review']
test['review']= tempstart
test['star'] =  test['star']  - 1
#train.columns = ['star', 'title','review']
        
dictionary = VDCNNDictionary()

trainset = TextDataset(train, dictionary, False, 1024)
testset = TextDataset(test, dictionary, False, 1024)


# In[13]:


trainloader = torch.utils.data.DataLoader(trainset, batch_size= 128,
                                          shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)


# In[14]:


# for i,data in enumerate(trainloader, 0):
#     print(i)

# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#trainset.__getitem__(0)
#bigtensor


# In[89]:



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


class KMaxPool(nn.Module):

    def __init__(self, k = None):
        super().__init__()
        self.k = k

    def forward(self, x):
        if self.k is None:
            time_steps = x.shape[2]
            self.k = time_steps//2
        kmax, kargmax = x.topk(self.k, dim=2)
        #kmax, kargmax = x.topk(self.k)
        return kmax

    
# def downsampling(pool_type, in_channel,out_channel):
# #     if pool_type == '1':
# # #         pool = nn.Sequential(
# # #             nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
# # #             nn.BatchNorm1d(out_channel))
# #         pool = 
# #     el
#     if pool_type == '2':
#         pool = KMaxPool()
#     elif pool_type == '3':
#         pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
# #     else:
# #         pool = None
#     return pool
    
#initial length
#s = 16
nclasses = 10#!!!
vocabulary_size = 1024
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace = True)
        
        
    def forward(self, x):
        shortcut = 'off'
        if shortcut == 'on':
            residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        if shortcut == 'on':
            out += residual
        
        return out

downsample_option = '1'#!!!
shortcut = 'off'
class VDCNN(nn.Module):
    def __init__(self):
        
        super(VDCNN, self).__init__()                                    
        self.downsample_option = downsample_option
        #assert self.downsample_option in ['1', '2', '3'],'error message'
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16, padding_idx=0)
        
        self.conv1 = nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1)

        first_conv_stride = 1
        block1_out = 128 
        block2_in = 128 
        block2_out = 256
        block3_in = 256
        block3_out = 512
        block4_in = 512
        
        if self.downsample_option == '1':
            first_conv_stride = 2
        if self.downsample_option == '2':
            self.downsample_2_1 = KMaxPool()
            self.downsample_2_2 = KMaxPool()
            self.downsample_2_3 = KMaxPool()
        if self.downsample_option == '3':
            self.downsample_3_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            self.downsample_3_2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            self.downsample_3_3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = nn.Sequential(
            conv_block(64,64,1),
            conv_block(64,64,1),
            conv_block(64,64,1),
            conv_block(64,block1_out,1))
        
        self.block2 = nn.Sequential(
            conv_block(block2_in,128,first_conv_stride),
            conv_block(128,128,1),
            conv_block(128,128,1),
            conv_block(128,block2_out,1))
        
        self.block3 = nn.Sequential(
            conv_block(block3_in,256,first_conv_stride),
            conv_block(256,256,1),
            conv_block(256,256,1),
            conv_block(256,block3_out,1))
        
        self.block4 = nn.Sequential(
            conv_block(block4_in,512,first_conv_stride),
            conv_block(512,512,1),
            conv_block(512,512,1),
            conv_block(512,512,1))
        
        self.kmaxpool = KMaxPool(8)
        self.linear1 = nn.Sequential(
            nn.Linear(4096,2048) ,nn.ReLU())
        
        self.linear2 = nn.Sequential(
            nn.Linear(2048,2048) ,nn.ReLU())
        
        self.linear3 = nn.Linear(2048,nclasses)
        
    def forward(self, x):
        #print('input',x.size())
        display_option = self.downsample_option
        
        assert display_option in ['1', '2', '3'],'downsampling option must be 1,2 or 3'
        x = self.embedding(x)
        #print('embed',x.size())
        x = x.transpose(1,2)
        x = x.transpose(0,2)
        
        x = self.conv1(x)
        #print('after conv1',x.size())
        #****************************************
        x = self.block1(x)
        #print('after block 1 ',x.size())
        if self.downsample_option == '2':
            x = self.downsample_2_1(x)
        elif self.downsample_option == '3':
            x = self.downsample_3_1(x)
        #print('after 1st pool/2',x.size())
        
        #****************************************
        
        x = self.block2(x)
        #print('after block2 ',x.size())
        if self.downsample_option == '2':
            x = self.downsample_2_2(x)
        elif self.downsample_option == '3':
            x = self.downsample_3_2(x)
        #print('after 2 pool ',x.size())
        #****************************************
        #print('before b3 ',x.size())
        x = self.block3(x)
        #print(x.size())
        if self.downsample_option == '2':
            x = self.downsample_2_3(x)
        elif self.downsample_option == '3':
            x = self.downsample_3_3(x)
            
        #****************************************    
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = self.kmaxpool(x)
        #print('before linear ',x.size())
        #x = kmax_pooling(x, dim=2, k=8)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        #print(x.size())
        return x


net = torch.load('yh_1_17_epoch1.model')
net.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.4)


# In[90]:


# Train

net.train()
print('downsample option: ',downsample_option)
for epoch in range(20):  # loop over the dataset multiple times
    time0=time.time()
    scheduler.step()
    net.train()
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        # get the inputs
        inputs,labels = data
        
        inputs = torch.transpose(inputs, 0, 1)
        inputs = inputs.to(device)
        
        #labels = torch.tensor(labels)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #torch.cuda.synchronize()
        loss.backward()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        optimizer.step()

        #print statistics
        running_loss += loss.item()
        if i % 1000 == 999 and (i != 1):    # print every 
            print('[%d, %5d] loss: %.3f time: %.4f' %
                 (epoch +1 + 1, i + 1, running_loss / 1000, time.time()-time0))
            running_loss = 0.0

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            
            inputs = torch.transpose(inputs, 0, 1)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch',(epoch+2),'Accuracy : %f %%' % (
        100 * correct / total))
    torch.save(net,'../model/yh_1_17_temp.model')
# model names: dataset _ downsampleing option _ layers
torch.save(net,'../model/yh_1_17_final.model')
print('Finished Training')



