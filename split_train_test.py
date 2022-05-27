import pandas as pd
import numpy as np
import torch

from const import *

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
    id_df = pd.read_parquet(id_file_name)
    
    save_df = main_df[main_df.id.isin(id_df.id)]
    
    data = save_df.apply(convert_tensor, axis=1).tolist()
    torch.save(data, save_file_name)
    
    
    
embed_final = pd.read_parquet(embedded_entube_path)

get_split_file(DATA_SAMPLE_DIR+'entube_train.parquet', 'entube_embedding_train.pt', embed_final)
get_split_file(DATA_SAMPLE_DIR+'entube_val.parquet', 'entube_embedding_val.pt', embed_final)
get_split_file(DATA_SAMPLE_DIR+'entube_test.parquet', 'entube_embedding_test.pt', embed_final)