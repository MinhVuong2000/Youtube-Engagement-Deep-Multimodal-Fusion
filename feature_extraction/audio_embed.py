#load data
import pandas as pd
import numpy as np

from ...const import *

print('Setup data entube')
entube = pd.read_parquet(ENTUBE_PATH)

# load model
print('Setup model VGGish')
import torch
from torch import nn

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.eval().to(device)

#load audio
print('Setup path audio')
format_path = DATA_SAMPLE_DIR + 'audio_by_year/{}/{}.wav'
entube['year_upload'] = entube['upload_date'].dt.year
audio_df = entube[['id', 'year_upload']]

audio_df = audio_df[audio_df.year_upload.isin([2016,2017,2020,2021])]


audio_df['path'] = audio_df.apply(lambda row: format_path.format(row.year_upload,row.id), axis=1)


AUDIO_EMBED_PATH = EMBEDDED_DATA_DIR + 'audio_embedding.parquet'

def embed_audio_handler(audio_path):
  try:
    embed = model.forward(audio_path)
    return embed.tolist()
  except Exception as e:
    print(f'Error at {audio_path}: {e}')
    return None

#embedding
print('Start Embedding...')
end = len(entube)
batch_size = 500
start=0

for idx in range(start, end, batch_size):
  audios_df_batch = audio_df.iloc[idx:idx+batch_size]
  audios_df_batch['embedding_audio'] = audios_df_batch['path'].apply(embed_audio_handler)
  if idx==0:
    audios_df_batch.to_parquet(AUDIO_EMBED_PATH)
  else:
    old_df = pd.read_parquet(AUDIO_EMBED_PATH)
    new_df = pd.concat([old_df, audios_df_batch], axis=0)
    new_df.to_parquet(AUDIO_EMBED_PATH)
  print(f'Index {idx}->{idx+batch_size} is Done')

print("All is done")
