############baseline############
#import packages
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

#define instances
TRAIN_EMBED_PATH = '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_train.pt'
VAL_EMBED_PATH = '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_val.pt'
TEST_EMBED_PATH = '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_test.pt'

SELECT_LABEL = 'label_2'
SAVING_FOLDER = '1fc'
CHECKPOINT_DIR = f'/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/checkpoints/{SELECT_LABEL}/{SAVING_FOLDER}'
writer = SummaryWriter(f'/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/logs/{SELECT_LABEL}/{SAVING_FOLDER}')

PATIENCE = 20
BATCH_SIZE = 64
NUM_EPOCH = 50

INPUT_SAME = 512
INPUT_TEXT = 768
INPUT_THUMBNAIL = 2560
INPUT_VIDEO = 2304*2*2
INPUT_AUDIO = 62*128


#define models
class ThesisBaseline(nn.Module):
  def __init__(self):
    super(ThesisBaseline, self).__init__()
    self.same_shape_title = nn.Sequential(nn.Linear(INPUT_TEXT, INPUT_SAME, bias=True), 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout(p = 0.1)
                                     )
    self.same_shape_tag = nn.Sequential(nn.Linear(INPUT_TEXT, INPUT_SAME, bias=True), 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout(p = 0.1)
                                     )
    self.same_shape_thumbnail = nn.Sequential(nn.Linear(INPUT_THUMBNAIL, INPUT_SAME, bias=True), 
                                         nn.ReLU(inplace=True), 
                                         nn.Dropout(p = 0.1)
                                         )
    self.same_shape_video = nn.Sequential(nn.Linear(INPUT_VIDEO, INPUT_SAME, bias=True), 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout(p = 0.15)
                                     )
    self.same_shape_audio = nn.Sequential(nn.Linear(INPUT_AUDIO, INPUT_SAME, bias=True), 
                                     nn.ReLU(inplace=True), 
                                     nn.Dropout(p = 0.15)
                                     )
    
    self.fc = nn.Linear(INPUT_SAME*5, 3)

  def forward(self, X):
    # embed: (input_title_embed, input_tag_embed, input_thumbnail_embed, input_video_embed, input_audio_embed)
    X_title, X_tag, X_thumbnail, X_video, X_audio = X
    # embedding
    x_title = self.same_shape_title(X_title)
    x_tag = self.same_shape_tag(X_tag)
    x_thumbnail = self.same_shape_thumbnail(X_thumbnail)
    x_video = X_video.view(X_video.size(0), -1)
    x_video = self.same_shape_video(x_video)
    x_audio = X_audio.view(X_audio.size(0), -1)
    x_audio = self.same_shape_audio(x_audio)

    # concat
    x = torch.cat((x_title, x_tag, x_thumbnail, x_video, x_audio), dim=1)

    # fully connected
    x = self.fc(x)
    return x
    
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
    
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model, epoch, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model the best...')
        torch.save(
              {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
               }, 
               os.path.join(CHECKPOINT_DIR, f'model_best_loss_val.pt')
            )
        self.val_loss_min = val_loss
    
# func train
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

def train_baseline(model, epochs, loss_fn, optimizer, train_loader, val_loader):
    len_train_loader = len(train_loader)
    len_val_loader = len(val_loader)
    model.train()
    early_stop = EarlyStopping(patience=PATIENCE, verbose=True, delta=0.001)
    for epoch in range(1, epochs+1):
        loss_train = 0.0
        pred_train = []
        lbl_train = []
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

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
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

train_dataset = EntubeDataset(train, device)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dataset = EntubeDataset(val, device)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

model = ThesisBaseline()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
print("Done init model")

# train
print("Start train ...")
train_baseline(model, NUM_EPOCH, loss_fn, optimizer, train_loader, val_loader)

print('Done Training')


#test
print('Start testing...')
test = torch.load(TEST_EMBED_PATH)
test_dataset = EntubeDataset(test, device)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, f'model_best_loss_val.pt'))
model = ThesisBaseline()
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

print("10-first lablel:",lbl_test[:10])
print("10-first predict:",pred_test[:10])

metrics = classification_report(lbl_test, pred_test)
print("Done Testing. Classification_report for testing:")
print(metrics)
      