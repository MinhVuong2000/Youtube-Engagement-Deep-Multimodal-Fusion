#load data
import pandas as pd
import numpy as np

print('Setup data entube')
ENTUBE = '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/entube.parquet'
entube = pd.read_parquet(ENTUBE)

# load model
print('Setup model Slowfast')
from typing import Dict
import os
import json
import urllib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
model = torch.nn.Sequential(*(list(model.modules())[1][:-1]))
device = "cuda"
model = model.eval()
model = model.to(device)

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
slowfast_alpha = 4
num_clips = 10
num_crops = 3

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),
            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)
clip_duration = (num_frames * sampling_rate) / frames_per_second

#load video
print('Setup path video')
format_path = '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/video_short_by_year/{}/{}.mp4'
entube['year_upload'] = entube['upload_date'].dt.year
video_df_full = entube[['id', 'year_upload']]

video_df = video_df_full[video_df_full.year_upload.isin([2019])].reset_index(drop=True)
del entube
del video_df_full

print(video_df.year_upload.value_counts())

video_df['path'] = video_df.apply(lambda row: format_path.format(row.year_upload,row.id), axis=1)
print(video_df.columns)

VIDEO_EMBED_PATH = '/home/lttung/EnTube/Data/content/drive/MyDrive/EnTube/data/DataFourCate/metadata/video_embedding_19.parquet'

def embed_video_handler(video_path):
    try:
        # Select the duration of the clip to load by specifying the start and end duration
        # The start_sec should correspond to where the action occurs in the video
        start_sec = 0
        end_sec = start_sec + clip_duration

        # Initialize an EncodedVideo helper class and load the video
        video = EncodedVideo.from_path(video_path)

        # Load the desired clip
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

        # Apply a transform to normalize the video input
        video_data = transform(video_data)

        # Move the inputs to the desired device
        inputs = video_data["video"]
        inputs = [i.to(device)[None, ...] for i in inputs]

        preds = model(inputs)[0].tolist()
        return preds
    except Exception as e:
        print(f"Error path {video_path}: {e}")
        return None

#embedding
print('Start Embedding...')
end = len(video_df)
batch_size = 500
start=0

print(video_df)

for idx in range(start, end, batch_size):
  videos_df_batch = video_df.iloc[idx:idx+batch_size]
  videos_df_batch['embedding_video'] = videos_df_batch['path'].apply(embed_video_handler)
  if idx==0:
    videos_df_batch.to_parquet(VIDEO_EMBED_PATH)
  else:
    old_df = pd.read_parquet(VIDEO_EMBED_PATH)
    new_df = pd.concat([old_df, videos_df_batch], axis=0)
    new_df.to_parquet(VIDEO_EMBED_PATH)
    del old_df
    del new_df
  del videos_df_batch
  print(f'Index {idx}->{idx+batch_size} is Done')

print("All is done")

# Failed to open video AEwDZBzdbi8.mp4. [Errno 1094995529] Invalid data found when processing input: '<none>'; last error log: [mov,mp4,m4a,3gp,3g2,mj2] moov atom not found