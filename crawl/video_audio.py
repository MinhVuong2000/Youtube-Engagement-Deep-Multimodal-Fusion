import os
import argparse
import shutil
from tqdm import tqdm

import yt_dlp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_extract_audio

from crawl.utils import load_data, VIDEO_AUDIO_COLUMN_STATUS, CrawlStatus


def crawler_video_audio(video_id, year, output_dir="../data"):
    try:
        video_url = "https://youtu.be/" + video_id
        # Tải xuống 1 phút đầu video + audio
        temp_video_path = os.path.join(output_dir+"/video/" + str(year) + "/tmp_" + str(video_id) + ".mp4")
        video_path = os.path.join(output_dir+"/video/" + str(year) + "/" + str(video_id) + ".mp4")
        download_opts = {
            "format": "mp4",
            "outtmpl": temp_video_path,
            "external_downloader": "ffmpeg",
            "external_downloader_args": [f"-ss 0 -i {temp_video_path} -t 60 -c copy {video_path}"]
        }
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([video_url])
        
        ffmpeg_extract_subclip(temp_video_path, 0, 60, outputfile=video_path)
        # os.system(f"ffmpeg -ss 0 -i {temp_video_path} -t 60 -c copy {video_path}")
        os.remove(temp_video_path)

        # Tách audio ra khỏi video
        audio_path = os.path.join(output_dir+"/audio/" + str(year) + "/" + str(video_id) + ".mp3")
        
        ffmpeg_extract_audio(video_path, audio_path)
        
        return True
    except Exception as e:
        print("❌ Không thể tải video")
        print(e)
        return False




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process category for video information.')
    parser.add_argument('--category', type=str, help='Category for video processing')
    args = parser.parse_args()
    category = args.category

    path = f'../data/status/video_audio_{category}.parquet'
    path_sample = '../data/status/{category}.parquet'

    try:
        df = load_data(category, path_sample=path_sample)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Copy file to status directory
        shutil.copy(f'../data/{category}.parquet', path)

        # Load data
        df = load_data(category, path_sample=path_sample)
        

    if VIDEO_AUDIO_COLUMN_STATUS not in df.columns:
        df[VIDEO_AUDIO_COLUMN_STATUS] = CrawlStatus.NOT_YET.value
        print(f"Created {VIDEO_AUDIO_COLUMN_STATUS}")

    for i in tqdm(range(len(df))):
        if df[VIDEO_AUDIO_COLUMN_STATUS][i] == CrawlStatus.DONE.value:
            continue

        success = crawler_video_audio(df["id"][i], category)
        if success:
            df.loc[i, VIDEO_AUDIO_COLUMN_STATUS] = CrawlStatus.DONE.value
        if (i % 100 == 10):
            print("Processing {}".format(i))
            df.to_parquet(path)

        df.to_parquet(path)
        break
