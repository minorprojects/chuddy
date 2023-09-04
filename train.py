import torch
import torch.nn.Functional as F
from torch import nn
from datasets import load_dataset
import transformers
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Models.chuddy import Chuddy
from utils import (
  lr,
  epochs,
  save_path,
  data_link,
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model = Chuddy()
model = model.to(device)
save_path= save_path
epochs=epochs

data = load_dataset(data_link)

data_loader = DataLoader(data,batched=True) 

def train(model,devivce,data_loader,epochs):
  for epoch in range(epochs):
  
    model.train()
    total_loss = 0.0
    for i,(image,text) in enumerate(data_loader):
      image = image.to(device)
      text = text.to(device)
      optimizer.zero_grad()
      loss_itc,loss_itm,loss_lm,loss_hyp = model(image,text)
      loss = loss_itc + loss_itm + loss_lm
      loss.backward()
      optimizer.step()
      total_loss = loss.item()
    print('epoch {} step {} loss: {}'.format(epoch,i + 1, total_loss))
return total_loss

torch.save({
  'epoch': epochs,
  model_state_dict()},save_path)
    
    

  
  
  
