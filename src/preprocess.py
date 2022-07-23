import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy.data import Field, BucketIterator, Iterator
from torchtext.legacy import data
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import spacy
import numpy as np
import pandas as pd
import random
import math
import time
from tokenize import tokenize, untokenize
import io
import keyword
from tqdm import tqdm

print("modules loaded for preprocess")


f = open("dataset/eng2py.txt", "r")
file_lines = f.readlines()


dps = []
dp = None
for line in file_lines:
  if line[0] == "#":
    if dp:
      dp['solution'] = ''.join(dp['solution'])
      dps.append(dp)
    dp = {"question": None, "solution": []}
    dp['question'] = line[1:]
  else:
    dp["solution"].append(line)

i=0
for dp in dps:
  print("\n Question no: ", i+1)
  i+=1
  print(dp['question'][1:])
  print(dp['solution'])
  if i>9:
    break

print("Dataset size:", len(dps))



def tokenize_python_code(python_code_str):
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []
    for i in range(0, len(python_tokens)):
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    return tokenized_output

tokenized_sample = tokenize_python_code(dps[1]['solution'])
print(tokenized_sample)

print(untokenize(tokenized_sample).decode('utf-8'))


print(keyword.kwlist)

def augment_tokenize_python_code(python_code_str, mask_factor=0.3):


    var_dict = {} # Dictionary that stores masked variables

    # certain reserved words that should not be treated as normal variables and
    # hence need to be skipped from our variable mask augmentations
    skip_list = ['range', 'enumerate', 'print', 'ord', 'int', 'float', 'zip'
                 'char', 'list', 'dict', 'tuple', 'set', 'len', 'sum', 'min', 'max']
    skip_list.extend(keyword.kwlist)

    var_counter = 1
    python_tokens = list(tokenize(io.BytesIO(python_code_str.encode('utf-8')).readline))
    tokenized_output = []

    for i in range(0, len(python_tokens)):
      if python_tokens[i].type == 1 and python_tokens[i].string not in skip_list:
        
        if i>0 and python_tokens[i-1].string in ['def', '.', 'import', 'raise', 'except', 'class']: # avoid masking modules, functions and error literals
          skip_list.append(python_tokens[i].string)
          tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
        elif python_tokens[i].string in var_dict:  # if variable is already masked
          tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
        elif random.uniform(0, 1) > 1-mask_factor: # randomly mask variables
          var_dict[python_tokens[i].string] = 'var_' + str(var_counter)
          var_counter+=1
          tokenized_output.append((python_tokens[i].type, var_dict[python_tokens[i].string]))
        else:
          skip_list.append(python_tokens[i].string)
          tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
      
      else:
        tokenized_output.append((python_tokens[i].type, python_tokens[i].string))
    
    return tokenized_output

tokenized_sample = augment_tokenize_python_code(dps[1]['solution'])
print(tokenized_sample)

print(untokenize(tokenized_sample).decode('utf-8'))

"""As one can see our augmented tokenizer picked num2 randomly and masked(replaced) it with by var_1

## Building Train and Validation Dataset
"""

python_problems_df = pd.DataFrame(dps)

python_problems_df.head()

python_problems_df.shape

import numpy as np

np.random.seed(0)
msk = np.random.rand(len(python_problems_df)) < 0.90 # Splitting data into 90% train and 10% validation

train_df = python_problems_df[msk]
val_df = python_problems_df[~msk]

train_df.shape

val_df.shape

"""## Creating vocabulary using torchtext

In this section we will use torchtext Fields to construct the vocabulary for our sequence-to-sequence learning problem.
"""

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

Input = data.Field(tokenize = 'spacy',
            init_token='<sos>', 
            eos_token='<eos>', 
            lower=True)

Output = data.Field(tokenize = augment_tokenize_python_code,
                    init_token='<sos>', 
                    eos_token='<eos>', 
                    lower=False)

fields = [('Input', Input),('Output', Output)]

"""Since our data augmentations have the potential to increase the vocabulary beyond what it initially is, we must ensure that we capture as many variations as possible in the vocabulary that we develop. In the the below code we apply our data augmentations 100 times to ensure that we can capture a majority of augmentations into our vocabulary. """

train_example = []
val_example = []

train_expansion_factor = 100
for j in range(train_expansion_factor):
  for i in range(train_df.shape[0]):
      try:
          ex = data.Example.fromlist([train_df.question[i], train_df.solution[i]], fields)
          train_example.append(ex)
      except:
          pass

for i in range(val_df.shape[0]):
    try:
        ex = data.Example.fromlist([val_df.question[i], val_df.solution[i]], fields)
        val_example.append(ex)
    except:
        pass

train_data = data.Dataset(train_example, fields)
valid_data =  data.Dataset(val_example, fields)

Input.build_vocab(train_data, min_freq = 0)
Output.build_vocab(train_data, min_freq = 0)

Output.vocab

def save_vocab(vocab, path):
    import pickle
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

train_data[1].Output

print(vars(train_data.examples[1]))

