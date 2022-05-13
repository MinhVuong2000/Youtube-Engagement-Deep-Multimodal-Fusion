import pandas as pd
import numpy as np
import torch

def convert_tensor(row):
    tensor_title = torch.tensor(row['embedding_title'], dtype=torch.float)
    tensor_tag = torch.tensor(row['embedding_tags'], dtype=torch.float)
    tensor_thumbnail = torch.tensor(row['embedding_thumbnail'], dtype=torch.float)
    tensor_video = torch.tensor(np.array(list(map(lambda x: list(map(list, x)),row['embedding_video']))), dtype=torch.float)
    tensor_audio = torch.tensor(np.array(list(map(list,row['embedding_audio']))), dtype=torch.float)
    
    lbl_1_tensor = torch.tensor(row['label_1'])
    lbl_2_tensor = torch.tensor(row['label_2'])
    
    return {
        'id': row.id,
        'embedding_title':tensor_title,
        'embedding_tag':tensor_tag,
        'embedding_thumbnail':tensor_thumbnail,
        'embedding_video':tensor_video,
        'embedding_audio':tensor_audio,
        'label_1':lbl_1_tensor,
        'label_2':lbl_2_tensor
    }

def get_split_file(id_file_name, save_file_name, main_df):
    id_df = pd.read_parquet("/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/{}.parquet".format(id_file_name))
    
    save_df = main_df[main_df.id.isin(id_df.id)]
    
    data = save_df.apply(convert_tensor, axis=1).tolist()
    #save_df.to_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/{}.parquet'.format(save_file_name))
    torch.save(data, '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/{}.pt'.format(save_file_name))
    
    
    
embed_final = pd.read_parquet('/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube_embedding_final_final.parquet')

# embed_final = embed_final[~(embed_final.embedding_video.isna() | embed_final.embedding_audio.isna())]

get_split_file('entube_train', 'entube_embedding_train', embed_final)
get_split_file('entube_val', 'entube_embedding_val', embed_final)
get_split_file('entube_test', 'entube_embedding_test', embed_final)