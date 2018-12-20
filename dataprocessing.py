
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

import nltk
import itertools
import io
#ttt = torch.load('ag_test_.pt')


# In[2]:


# 
#ag1 = './datasets/ag_news_csv/train.csv'
#ag2 = './datasets/ag_news_csv/test.csv'
db1 = 'E:/study/CS598/project/dbpedia_csv/train.csv'
db2 = 'E:/study/CS598/project/dbpedia_csv/test.csv'
train = pd.read_csv(db1, header=None,usecols = [0,1,2],parse_dates = [[1,2]] )

train.columns = ['star', 'review']
tempstar = train['star']
train['star']=train['review']
train['review']= tempstar
train['star'] =  train['star']  - 1


test = pd.read_csv(db2, header=None,usecols = [0,1,2],parse_dates=[[1,2]] )
test.columns = ['star', 'review']
tempstart = test['star']
test['star']=test['review']
test['review']= tempstart
test['star'] =  test['star']  - 1


# In[4]:


start = time.time()
train['review'] = train['review'].apply(nltk.word_tokenize)
print ("train set series.apply", (time.time() - start))


# In[5]:


start = time.time()
test['review'] = test['review'].apply(nltk.word_tokenize)
print ("test set series.apply", (time.time() - start))


# In[8]:



torch.save(train,'db_train_token.pt')
torch.save(test,'db_test_token.pt')


# In[3]:


# tr=torch.load('ypf_train_token.pt')
# te=torch.load('ypf_test_token.pt')

tr=torch.load('E:/study/CS598/project/db_train_token.pt')
te=torch.load('E:/study/CS598/project/db_test_token.pt')


# In[4]:


glove_dictionary = np.load('glove_dictionary.npy')
glove_embeddings = np.load('glove_embeddings.npy')
word_to_id = {token: idx for idx, token in enumerate(glove_dictionary)}

x_train_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in tr['review']]
x_test_token_ids = [[word_to_id.get(token,-1)+1 for token in x] for x in te['review']]


# In[ ]:


# i = 0

# for text in x_train_token_ids:
#     train['review'][i] = text
#     i = i+1
#     if i%1000 ==0 :
#         print(i)
# i=0
# for text in x_test_token_ids:
#     test['review'][i] = text
#     i = i+1
#     if i%1000 ==0 :
#         print(i)
    


# In[11]:



# i = 0

# for text in x_train_token_ids:
#     line = glove_embeddings[x_train_token_ids[i]]
#     avg = np.mean(line,axis = 1)
#     train['review'][i] = line
#     i = i+1
#     if i%1000 ==0 :
#         print(i)
# i=0
# for text in x_test_token_ids:
#     line = glove_embeddings[x_test_token_ids[i]]
#     avg = np.mean(line,axis = 1)
#     test['review'][i] = line
#     i = i+1
#     if i%1000 ==0 :
#         print(i)
    
    

# torch.save(train,'ag_train_tokenid.pt')
# torch.save(test,'ag_test_tokenid.pt')

