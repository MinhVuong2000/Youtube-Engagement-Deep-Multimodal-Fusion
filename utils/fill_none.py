# import pandas as pd

# video = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/video_embedding_16_21.parquet').drop_duplicates(subset=['id']).reset_index(drop=True)
# audio = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/audio_embedding_16_17_20_21.parquet').drop_duplicates(subset=['id']).reset_index(drop=True)
# entube = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_label.parquet').drop_duplicates(subset=['id']).reset_index(drop=True)
# id_final = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_final.parquet')

# entube = entube[entube.id.isin(id_final.id)].reset_index(drop=True)

# def fill_none(r, df2, embedding_col):
  # try:
    # return df2[df2['id']==r['id']][embedding_col].iloc[0]
  # except:
    # return r[embedding_col]
    
# entube['embedding_video'] = entube.apply(lambda r: fill_none(r, video, 'embedding_video'), axis=1)
# entube['embedding_audio'] = entube.apply(lambda r: fill_none(r, audio, 'embedding_audio'), axis=1)

# entube.to_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_final.parquet')

import pandas as pd

video = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/video_embedding_19.parquet').drop_duplicates(subset=['id']).reset_index(drop=True)
entube = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_final.parquet')

def fill_none(r, df2, embedding_col):
  try:
    return df2[df2['id']==r['id']][embedding_col].iloc[0]
  except:
    return r[embedding_col]
    
entube['embedding_video'] = entube.apply(lambda r: fill_none(r, video, 'embedding_video'), axis=1)

entube.to_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_final_final.parquet')