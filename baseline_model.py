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

from const import *

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
    