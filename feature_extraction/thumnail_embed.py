import torch
from torchvision import models
import numpy as np
import pandas as pd
import cv2
import warnings
from typing import List, Optional

def setup_model(device: torch.device) -> torch.nn.Module:
    efficientnet = models.efficientnet_b7(pretrained=True)
    efficientnet = torch.nn.Sequential(*(list(efficientnet.children())[:-1]))
    return efficientnet.eval().to(device)

def setup_dataframe(entube: pd.DataFrame, data_sample_dir: str) -> pd.DataFrame:
    path_thumbnail = f"{data_sample_dir}thumbnails_by_year/{{0}}/{{1}}_{{2}}.jpg"
    entube['year_upload'] = entube['upload_date'].dt.year
    thumbnail_df = entube[['id', 'year_upload']]
    for i in range(1, 4):
        thumbnail_df[f'path_{i}'] = thumbnail_df.apply(lambda row: path_thumbnail.format(row.year_upload, row.id, i), axis=1)
    return thumbnail_df

def embed_thumbnail(paths: List[str], model: torch.nn.Module, device: torch.device = torch.device('cuda')) -> Optional[List[float]]:
    size = (224, 224, 3)
    temp = [cv2.resize(cv2.imread(path), size[:2]) for path in paths if cv2.imread(path) is not None]
    if not temp:
        return None
    im_tensor = torch.tensor(temp).permute(0, 3, 1, 2) / 255.0
    im_tensor = im_tensor.to(device)
    embed = model(im_tensor).squeeze()
    return embed.mean(0).tolist()

def process_thumbnails(thumbnail_df: pd.DataFrame, model: torch.nn.Module, device: torch.device, start: int, end: int, batch_size: int, embedded_data_dir: str) -> None:
    thumbnail_embed_path = f"{embedded_data_dir}thumbnail_embedding.parquet"
    for idx in range(start, end, batch_size):
        thumbnails_df_batch = thumbnail_df.iloc[idx:idx+batch_size]
        thumbnails_df_batch['embedding_thumbnail'] = thumbnails_df_batch.apply(
            lambda row: embed_thumbnail([row[f'path_{i}'] for i in range(1, 4)], model, device), axis=1
        )
        if idx == 0:
            thumbnails_df_batch.to_parquet(thumbnail_embed_path)
        else:
            old_df = pd.read_parquet(thumbnail_embed_path)
            new_df = pd.concat([old_df, thumbnails_df_batch], axis=0)
            new_df.to_parquet(thumbnail_embed_path)
        print(f'Index {idx} is Done')
    print('All is Done!')

def main(entube_path: str, embedded_data_dir: str, data_sample_dir: str, start: int, end: Optional[int], batch_size: int) -> None:
    warnings.filterwarnings('ignore')
    
    entube = pd.read_parquet(entube_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for inference')
    
    model = setup_model(device)
    thumbnail_df = setup_dataframe(entube, data_sample_dir)
    
    if end is None:
        end = len(entube)
    
    process_thumbnails(thumbnail_df, model, device, start, end, batch_size, embedded_data_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thumbnail embedding script')
    parser.add_argument('--entube_path', type=str, required=True, help='Path to entube.parquet')
    parser.add_argument('--embedded_data_dir', type=str, required=True, help='Path to embedded data directory')
    parser.add_argument('--data_sample_dir', type=str, required=True, help='Path to data samples directory')
    parser.add_argument('--start', type=int, default=0, help='Start index for processing')
    parser.add_argument('--end', type=int, default=None, help='End index for processing')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing')
    args = parser.parse_args()
    
    main(args.entube_path, args.embedded_data_dir, args.data_sample_dir, args.start, args.end, args.batch_size)
