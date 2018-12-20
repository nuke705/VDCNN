
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
import time
import os
import pandas as pd
from os.path import exists

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def pad_text(text, pad, fixed_length):
    length = len(text)
    if length < fixed_length:
        return text + [pad]*(fixed_length - length)
    else:
        return text[:fixed_length]


class TextDataset(Dataset):
    
    def __init__(self, texts, sort=False):
        self.texts = texts.values.tolist()
              
    def __getitem__(self, index):
        tokens, label = self.texts[index]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
    def __len__(self):
        return len(self.texts)


# In[2]:


# In[ ]:
trainset = torch.load('./datasets/db_train_token.pt')
testset = torch.load('./datasets/db_test_token.pt')

#load tokenized files
#trainset = torch.load('E:/study/CS598/project/ag_train_token.pt')
#testset = torch.load('E:/study/CS598/project/ag_test_token.pt')


# In[ ]:


#optional: use glove dictionary to get token ids (not using embedding):
glove_dictionary = np.load('./datasets/glove_dictionary.npy')
#glove_embeddings = np.load('glove_embeddings.npy')
word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

# add a placeholder token
PAD_TOKEN = '#NAN'
word_to_id[PAD_TOKEN] = len(word_to_id) + 1

trainset_pad = list(map(lambda x: pad_text(x, PAD_TOKEN, 200), trainset['review'].tolist()))
testset_pad = list(map(lambda x: pad_text(x, PAD_TOKEN, 200), testset['review'].tolist()))


# In[3]:


x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in trainset_pad]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in testset_pad]

trainset['review'] = x_train_token_ids
testset['review'] = x_test_token_ids
# trainset['review'] = trainset_pad
# testset['review'] = testset_pad


# In[4]:


# In[ ]:


#set up dataset calss 
trainset = TextDataset(trainset, False)
testset = TextDataset(testset, False)
testset


# In[5]:


trainloader = torch.utils.data.DataLoader(trainset, batch_size= 100,
                                         shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)
# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[6]:



from RNN_model import LSTM_model
#def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=True, embedding_tensor=None,
#                 padding_index=0, hidden_size=64, num_layers=1, batch_first=True):

nclasses = 4  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# you can change input size for embedding layer
vocab_size = 100001
net = LSTM_model(vocab_size,300,nclasses)
net.to(device)


# In[7]:


# In[ ]:


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.RMSprop(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8,   12,      16 ], gamma=0.1)


# Train
time1 = time.time()
net.train()
#print('downsample option: ',downsample_option)
for epoch in range(30):  # loop over the dataset multiple times
    scheduler.step()
    net.train()
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        # get the inputs
        labels,inputs = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)

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
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for labels,inputs in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch',epoch,'Accuracy : %f %%' % (
        100 * correct / total))
    torch.save(net,'db_lstm.model')
time2 = time.time()
print(time2-time1)
# model names: dataset _ downsampleing option _ layers
torch.save(net,'db_lstm.model')
print('Finished Training')

