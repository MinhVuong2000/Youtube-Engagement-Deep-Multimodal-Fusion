import argparse
import shutil
from tqdm import tqdm

import yt_dlp

from crawl.utils import load_data, TEXT_COLUMN_STATUS, CrawlStatus


def get_title_tags(video_id):
    video_url = f"https://youtu.be/{video_id}"
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
        'write_title': True,
        'write_tags': True,
        'cookiesfrombrowser': "chrome:~/.var/app/com.google.Chrome/",
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=False)
        
    title = info.get('title', '')
    tags = info.get('tags', [])[:5]  # Get top 5 tags
    
    return {"title": title, "tags": tags}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process category for video information.')
    parser.add_argument('--category', type=str, help='Category for video processing')
    args = parser.parse_args()
    category = args.category

    path = f'../data/status/text_{category}.parquet'
    path_sample = '../data/status/text_{category}.parquet'

    try:
        df = load_data(category, path_sample=path_sample)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Copy file to status directory
        shutil.copy(f'../data/{category}.parquet', path)

        # Load data
        df = load_data(category, path_sample=path_sample)
        

    if TEXT_COLUMN_STATUS not in df.columns:
        df[TEXT_COLUMN_STATUS] = CrawlStatus.NOT_YET.value
        print(f"Created {TEXT_COLUMN_STATUS}")

    for i in tqdm(range(len(df))):
        if df[TEXT_COLUMN_STATUS][i] == CrawlStatus.DONE.value:
            continue
        try:
            result = get_title_tags(df["id"][i])
            df.loc[i, 'title'] = result['title']
            df.loc[i, 'tags'] = ",".join(result['tags'])
            df.loc[i, TEXT_COLUMN_STATUS] = CrawlStatus.DONE.value
        except Exception as e:
            print(f"Error processing video {df['id'][i]}: {e}")
        if (i % 100 == 10):
            print("Processing {}".format(i))
            df.to_parquet(path)
