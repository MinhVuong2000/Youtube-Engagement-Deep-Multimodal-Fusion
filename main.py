
#import packages
import os
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# func train
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
# import built class
from const import *
from early_stopping import EarlyStopping
from improved_model import ThesisEngagement

writer = SummaryWriter(LOG_DIR)

class EntubeDataset(Dataset):
  def __init__(self, data, device='cuda'):
        self.data = data
        self.device = device

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data[idx]
    lbl_tensor = data[SELECT_LABEL].to(self.device)

    tensor_title = data['embedding_title'].to(self.device)
    tensor_tag = data['embedding_tag'].to(self.device)
    tensor_thumbnail = data['embedding_thumbnail'].to(self.device)
    tensor_video = data['embedding_video'].to(self.device)
    tensor_audio = data['embedding_audio'].to(self.device)

    res = ((tensor_title, tensor_tag, tensor_thumbnail, tensor_video, tensor_audio), lbl_tensor)
    return res


def train_model(model, epochs, loss_fn, optimizer, train_loader, val_loader):
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
        
    if os.path.exists(CHECKPOINT_DIR):
        print('Delete old check point')
        os.system(f'rm -rf {CHECKPOINT_DIR}')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    early_stop = EarlyStopping(patience=PATIENCE, verbose=True, delta=0.001)
    for epoch in range(1, epochs+1):
        loss_train = 0.0
        pred_train = []
        lbl_train = []
        model.train()
        loop = tqdm(train_loader, total = len_train_loader)
        loop.set_description(f"Epoch [{epoch}/{epochs}]")
        for embeds, labels in train_loader: 
            outputs = model(embeds)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            lbl_train.append(labels.cpu())
            _, predicts = torch.max(outputs, 1)
            pred_train.append(predicts.cpu())
            loop.update(1)
            loop.set_postfix(loss_train_batch='{:.4f}'.format(loss.item()))
            
        loss_val = 0.0
        pred_val = []
        lbl_val = []
        model.eval()
        with torch.no_grad():
            for embeds, labels in val_loader:
                outputs = model(embeds)
                loss = loss_fn(outputs, labels)
                loss_val += loss.item()
                lbl_val.append(labels.cpu())
                _, predicts = torch.max(outputs, 1)
                pred_val.append(predicts.cpu())
                loop.set_postfix(loss_val_batch=loss.item())

        lbl_train = torch.cat(lbl_train, dim=0).numpy()
        pred_train = torch.cat(pred_train, dim=0).numpy()
        lbl_val = torch.cat(lbl_val, dim=0).numpy()
        pred_val = torch.cat(pred_val, dim=0).numpy()

        loss_train = loss_train/len_train_loader
        loss_val = loss_val/len_val_loader
        f1_train = f1_score(lbl_train, pred_train, average='micro')
        f1_val = f1_score(lbl_val, pred_val, average='micro')

        loop.set_postfix({
            'loss_train':'{:.4f}'.format(loss_train),
            'loss_val':'{:.4f}'.format(loss_val),
            'f1_train':'{:.4f}'.format(f1_train),
            'f1_val':'{:.4f}'.format(f1_val),
        })
        loop.close()

        writer.add_scalars("Loss", {'train':loss_train,
                                'val':loss_val}
                       ,epoch)
        writer.add_scalars("F1", {'train':f1_train,
                                'val':f1_val}
                      , epoch)
        
        #EarlyStopping and Save the model checkpoints 
        early_stop(loss_val, model, epoch, optimizer)
        if early_stop.early_stop==True:
            print(f'--------with patience={PATIENCE}, EarlyStopping at epoch : {epoch}')
            break
        else:
            torch.save(
              {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_train': loss_train,
                'loss_val': loss_val,
                'f1_train': f1_train,
                'f1_val': f1_val
               }, 
               os.path.join(CHECKPOINT_DIR, f'model_epoch{epoch}.pt')
            )   
print("Done define model")
      
#load data
train = torch.load(TRAIN_EMBED_PATH)
val = torch.load(VAL_EMBED_PATH)
print("Done load data")

#init to prepare train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# convert data to Dataset
train_dataset = EntubeDataset(train, device)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
val_dataset = EntubeDataset(val, device)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

model = ThesisEngagement()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("Done init model")

# train
print("Start train ...")
train_model(model, NUM_EPOCH, loss_fn, optimizer, train_loader, val_loader)
print('Done Training')

#test
print('Start testing...')
test = torch.load(TEST_EMBED_PATH)
test_dataset = EntubeDataset(test, device)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)

print('F1_score on test dataset of each epoch: ')
list_model = os.listdir(CHECKPOINT_DIR)
list_model.sort()
max_f1_test = 0
for path in list_model:
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, path))
    model = ThesisEngagement()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.eval().to(device)
    pred_test = []
    lbl_test = []
    with torch.no_grad():
        for embeds, labels in test_loader:
          outputs = model(embeds)
          lbl_test.append(labels.cpu())
          _, predicts = torch.max(outputs, 1)
          pred_test.append(predicts.cpu())
      
    lbl_test = torch.cat(lbl_test, dim=0).numpy()
    pred_test = torch.cat(pred_test, dim=0).numpy()

    # metrics = classification_report(lbl_test, pred_test)
    #print("Done Testing. Classification_report for testing:")
    f1 =  f1_score(lbl_test, pred_test, average='micro')
    if f1 > max_f1_test:
        max_f1_test = f1
    print(f'{path}: {round(f1,4)}')
    
print('Max f1 can get: ', round(max_f1_test,4))
print('Done Testing')