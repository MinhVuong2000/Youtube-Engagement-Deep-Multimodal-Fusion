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
from functools import reduce

from const import *

#define models
class View(nn.Module):
    def __init__(self, reshape):
        super().__init__()
        self.shape = reshape

    def __repr__(self):
        return f'View {self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


class ThesisEngagement(nn.Module):
  def __init__(self):
    super(ThesisEngagement, self).__init__()
    self.encoder_video_1 = nn.Sequential(
        nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1), padding=(0, 0, 0)),
        View((1,2304)),
    )
    self.encoder_video_2 = nn.Sequential(
        nn.LayerNorm((1,2304), eps=1e-05, elementwise_affine=True),
        MultiheadAttention(input_dim=2304, embed_dim=2304, num_heads=1),
        nn.LeakyReLU(inplace=True)
    )
    self.encoder_video_3 = nn.Sequential(
        View((2304,1)),
        nn.LayerNorm((2304,1), eps=1e-05, elementwise_affine=True),
        nn.Conv1d(2304, 1024, 1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        View((1,1024))
    )
    
    self.encoder_audio_1 = nn.Sequential(
        nn.LayerNorm((62,128), eps=1e-05, elementwise_affine=True),
        MultiheadAttention(input_dim=128, embed_dim=128, num_heads=1),
        nn.LeakyReLU(inplace=True)
    )
    self.encoder_audio_2 = nn.Sequential(
        nn.LayerNorm((62,128), eps=1e-05, elementwise_affine=True),
        nn.Conv1d(62, 8, 1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        View((1,1024))
    )
    
    self.view_title = View((1, 768))
    self.encoder_title = nn.Sequential(
        nn.LayerNorm((1,768), eps=1e-05, elementwise_affine=True),
        MultiheadAttention(input_dim=768, embed_dim=768, num_heads=1)
    )
    self.linear_title = nn.Linear(768, 1024)
    
    self.view_tag = View((1, 768))
    self.encoder_tag = nn.Sequential(
        nn.LayerNorm((1,768), eps=1e-05, elementwise_affine=True),
        MultiheadAttention(input_dim=768, embed_dim=768, num_heads=1)
    )
    self.linear_tag = nn.Linear(768, 1024)
    
    self.view_thumbnail = View((1, 2560))
    self.encoder_thumbnail = nn.Sequential(
        nn.LayerNorm((1,2560), eps=1e-05, elementwise_affine=True),
        MultiheadAttention(input_dim=2560, embed_dim=2560, num_heads=1)
    )
    self.linear_thumbnail = nn.Linear(2560, 1024)

    self.conv = nn.Sequential(
        nn.BatchNorm1d(4),
        nn.Conv1d(4, 1, 1),
        nn.ReLU(inplace=True)
    )
    self.cf = nn.Linear(1024,3)

  def forward(self, X):
    # unstack embed: (input_title_embed, input_tag_embed, input_thumbnail_embed, input_video_embed, input_audio_embed)
    X_title, X_tag, X_thumbnail, X_video, X_audio = X
    
    #encoder & context gating layer monomodal
    x_video_1 = self.encoder_video_1(X_video)
    x_video_2 = self.encoder_video_2(x_video_1)
    x_video = x_video_1 + x_video_2
    x_video = self.encoder_video_3(x_video)
    
    x_audio_1 = self.encoder_audio_1(X_audio)
    x_audio = X_audio + x_audio_1
    x_audio = self.encoder_audio_2(x_audio)
    
    x_title = self.view_title(X_title)
    x_title_1 = self.encoder_title(x_title)
    x_title = x_title + x_title_1
    x_title = self.linear_title(x_title)
    
    x_tag = self.view_tag(X_tag)
    x_tag_1 = self.encoder_tag(x_tag)
    x_tag = x_tag + x_tag_1
    x_tag = self.linear_tag(x_tag)
    
    x_thumbnail = self.view_thumbnail(X_thumbnail)
    x_thumbnail_1 = self.encoder_thumbnail(x_thumbnail)
    x_thumbnail = x_thumbnail + x_thumbnail_1
    x_thumbnail = self.linear_thumbnail(x_thumbnail)

    # fusion
    x_1 = torch.concat((x_title, x_tag, x_thumbnail), dim=1)#output shape: batch_size, 3, 1024
    x_2 = torch.max(torch.stack([x_audio, x_video]), dim=0)[0]
    x = x_1 + x_2

    # others fusion
    # x = torch.concat((x_title, x_tag, x_thumbnail), dim=1)
    # x = reduce(torch.mul, (x_title, x_tag, x_thumbnail))
    # x = torch.sum(torch.stack([x_title, x_tag, x_thumbnail]), dim=0)
    # x = torch.max(torch.stack([x_title, x_tag, x_thumbnail]), dim=0)[0]
    
    x = self.conv(x)
    x = x.view(x.size(0),1024)

    # fully connected
    x = F.dropout(x, 0.1)
    x = self.cf(x)
    return x
    