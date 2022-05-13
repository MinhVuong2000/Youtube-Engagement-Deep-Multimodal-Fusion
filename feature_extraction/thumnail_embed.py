import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
ENTUBE = '/home/s1820435/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube.parquet'
entube = pd.read_parquet(ENTUBE)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

efficientnet = models.efficientnet_b7(pretrained=True)
efficientnet = torch.nn.Sequential(*(list(efficientnet.children())[:-1]))
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
efficientnet.eval().to(device)

PATH_THUMBNAIL ='/home/s1820435/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/thumbnails_by_year/{}/{}_{}.jpg'
entube['year_upload'] = entube['upload_date'].dt.year
thumbnail_df = entube[['id', 'year_upload']]
thumbnail_df['path_1'] = thumbnail_df.apply(lambda row: PATH_THUMBNAIL.format(row.year_upload,row.id, 1), axis=1)
thumbnail_df['path_2'] = thumbnail_df.apply(lambda row: PATH_THUMBNAIL.format(row.year_upload,row.id, 2), axis=1)
thumbnail_df['path_3'] = thumbnail_df.apply(lambda row: PATH_THUMBNAIL.format(row.year_upload,row.id, 3), axis=1)
thumbnail_df.iloc[10].path_1

THUMBNAIL_EMBED_PATH = '/home/s1820435/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/thumbnail_embedding.parquet'

import cv2

SIZE = (224, 224, 3)

def embed_thumbnail(paths, device='cuda'):
  temp = []
  for path in paths:
    try:
        im = cv2.imread(path)
        im.resize(SIZE)
        temp.append(im)
    except:
        # error because file is not exist
        pass
  if not temp:
    return None
  im_tensor = torch.tensor(temp).permute(0, 3, 1, 2) / 255.0
  # im_tensor = im_tensor[[[2,1,0]]]
  im_tensor = im_tensor.to(device)
  embed = efficientnet(im_tensor).squeeze()
  return embed.mean(0).tolist()
  
start = 1000
end = len(entube)
batch_size=1000
for idx in range(start, end, batch_size):
  thumbnails_df_batch = thumbnail_df.iloc[idx:idx+batch_size]
  thumbnails_df_batch['embedding_thumbnail'] = thumbnails_df_batch.apply(
      lambda row: embed_thumbnail([row['path_1'], row['path_2'], row['path_3']], device), axis=1
  )
  if idx==0:
    thumbnails_df_batch.to_parquet(THUMBNAIL_EMBED_PATH)
  else:
    old_df = pd.read_parquet(THUMBNAIL_EMBED_PATH)
    new_df = pd.concat([old_df, thumbnails_df_batch], axis=0)
    new_df.to_parquet(THUMBNAIL_EMBED_PATH)
  print(f'Index {idx} is Done')
print('All is Done!')