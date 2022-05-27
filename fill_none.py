import pandas as pd
from const import *

video = pd.read_parquet(embedded_video_path).drop_duplicates(subset=['id']).reset_index(drop=True)
audio = pd.read_parquet(embedded_audio_path).drop_duplicates(subset=['id']).reset_index(drop=True)\
entube = pd.read_parquet(ENTUBE_PATH)

entube = entube[entube.id.isin(id_final.id)].reset_index(drop=True)

def fill_none(r, df2, embedding_col):
  try:
    return df2[df2['id']==r['id']][embedding_col].iloc[0]
  except:
    return r[embedding_col]
    
entube['embedding_video'] = entube.apply(lambda r: fill_none(r, video, 'embedding_video'), axis=1)
entube['embedding_audio'] = entube.apply(lambda r: fill_none(r, audio, 'embedding_audio'), axis=1)

entube.to_parquet(embedded_entube_path)
