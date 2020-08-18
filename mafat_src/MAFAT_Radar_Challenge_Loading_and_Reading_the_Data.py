#!/usr/bin/env python
# coding: utf-8

# #**MAFAT Radar Challenge - Loading and Reading the Data**
# 
# This notebook presents:
# 1. How to load the data.   
# 2. How to transform the Data into a Matlab file.
# 
# [Competition website](https://competitions.codalab.org/competitions/25389)   
# [MAFAT Challenge homepage](https://mafatchallenge.mod.gov.il/)

# *To load the data from Google Drive the data must be first uploaded to  Google Drive.*

# In[1]:


import pickle
import os
import pandas as pd
import numpy as np

from termcolor import colored


# In[2]:


#from google.colab import drive
#mount_path = '/content/gdrive'
#drive.mount(mount_path)


# In[ ]:


# Set and test path to competition data files
competition_path = 'INSERT HERE'
try:
  if competition_path == 'INSERT HERE':
    print('Please enter path to competition data files:')
    competition_path = input()
  file_path = 'MAFAT RADAR Challenge - Training Set V1.csv'
  with open(f'{mount_path}/{competition_path}/{file_path}') as f:
    f.readlines()
  print(colored('Everything is setup correctly', color='green'))
except:
  print(colored('Please mount drive and set competition_path correctly',
                color='red'))


# In[10]:


def load_data(file_path):
  """
  Reads all data files (metadata and signal matrix data) as python dictionary,
  the pkl and csv files must have the same file name.

  Arguments:
    file_path -- {str} -- path to the iq_matrix file and metadata file

  Returns:
    Python dictionary
  """
  pkl = load_pkl_data(file_path)
  meta = load_csv_metadata(file_path)
  data_dictionary = {**meta, **pkl}
  
  for key in data_dictionary.keys():
    data_dictionary[key] = np.array(data_dictionary[key])

  return data_dictionary


def load_pkl_data(file_path):
  """
  Reads pickle file as a python dictionary (only Signal data).

  Arguments:
    file_path -- {str} -- path to pickle iq_matrix file

  Returns:
    Python dictionary
  """
  path = os.path.join(file_path + '.pkl')
  with open(path, 'rb') as data:
    output = pickle.load(data)
  return output


def load_csv_metadata(file_path):
  """
  Reads csv as pandas DataFrame (only Metadata).

  Arguments:
    file_path -- {str} -- path to csv metadata file

  Returns:
    Pandas DataFarme
  """
  path = os.path.join(file_path + '.csv')
  with open(path, 'rb') as data:
    output = pd.read_csv(data)
  return output


# Below an example for loading all the training data and the public test data (signal + metadata).   
# It is possible to load only the csv metadata file or the signal pkl file by using the    
# "load_csv_metadata()" function or "load_pkl_data()" function.

# In[11]:


#train_path = 'MAFAT RADAR Challenge - Training Set V1'
#test_path = 'MAFAT RADAR Challenge - Public Test Set V1'

#training = load_data(train_path)
#test = load_data(test_path)


# In[12]:


train_path = '/home/vaibhawvipul/Documents/vaibhawvipul/datasets/mafat/MAFAT RADAR Challenge - Training Set V1'
test_path = '/home/vaibhawvipul/Documents/vaibhawvipul/datasets/mafat/MAFAT RADAR Challenge - Public Test Set V1'

training = load_data(train_path)
test = load_data(test_path)


# After the loading is finished the data is stored as a simple python dictionary.   
#    
# For more information on the data fields please go to the [competition website](https://competitions.codalab.org/competitions/25389)   
#  or read the [descriptive statistics](https://colab.research.google.com/drive/11Lzihg2vKIbo4KAIIJxW5CRZIncoWgtL?usp=sharing) notebook. 
# 
# 
# 

# In[13]:


list(training.keys())


# In[14]:


list(test.keys())


# ##Convert into MATLAB
# *This step is optional and useful for participants who want to use MATLAB to create model development.*
# 
# After loading the data it is possible to transform it into a mat file for MATLAB

# In[ ]:


from scipy.io import savemat  # Only for participants who wish to work on MATLAB

# Define the path for saving the *.mat file
training_mat_path = os.path.join(mount_path, competition_path, 
                                 train_path + '.mat')
test_mat_path = os.path.join(mount_path, competition_path, 
                             test_path + '.mat')

# savemat receives two arguments: saving path and dictionary 
savemat(training_mat_path, training)
savemat(test_mat_path, test)


# In[ ]:


# Download the mat files
from google.colab import files
files.download(training_mat_path)
files.download(test_mat_path)

