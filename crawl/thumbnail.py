import os
import requests
import argparse
import shutil
from tqdm import tqdm

import yt_dlp

from crawl.utils import load_data, THUMBNAIL_COLUMN_STATUS, CrawlStatus



def crawler_thumbnail(video_id, year, output_dir="../data"):
    video_url = "https://youtu.be/" + video_id
    try:
        # Tạo thư mục thumbnail nếu chưa có
        save_dir = os.path.join(output_dir, "thumbnail", year)
        os.makedirs(save_dir, exist_ok=True)

        # Cấu hình yt-dlp để thu thập thông tin video
        ydl_opts = {
            "quiet": True,
            "extract_flat": False,
            "nocheckcertificate": True,
            "format": "bestaudio/best",
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"},
            ],
            "outtmpl": {"default": os.path.join(output_dir, "%(id)s.%(ext)s")},
            "writethumbnail": True,  # Chỉ viết metadata thumbnail, không tải
            "skip_download": True,   # Không tải audio/video
        }

        # Thu thập thông tin video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        # ID video thực tế
        video_id = info["id"]

        # Lọc danh sách thumbnail
        thumbnail_urls = sorted(
            [thumb for thumb in info.get("thumbnails", []) if "height" in thumb],
            key=lambda x: x["height"], reverse=True
        )[:3]

        # Nếu không có height, lấy 3 ảnh đầu tiên
        if not thumbnail_urls:
            thumbnail_urls = info.get("thumbnails", [])[:3]

        # Tải xuống các thumbnail
        for i, thumb in enumerate(thumbnail_urls):
            thumb_url = thumb["url"]
            response = requests.get(thumb_url, stream=True)
            if response.status_code == 200:
                file_path = os.path.join(save_dir, f"{video_id}_thumb{i+1}.jpg")
                with open(file_path, "wb") as file:
                    file.write(response.content)
                print(f"✅ Đã tải {file_path}")
            else:
                print(f"❌ Không thể tải {thumb_url}")
        return True

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process category for video information.')
    parser.add_argument('--category', type=str, help='Category for video processing')
    args = parser.parse_args()
    category = args.category
        
    path = f'../data/status/thumbnail_{category}.parquet'
    path_sample = '../data/status/{category}.parquet'

    try:
        df = load_data(category, path_sample=path_sample)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Copy file to status directory
        shutil.copy(f'../data/{category}.parquet', path)

        # Load data
        df = load_data(category, path_sample=path_sample)
        

    if THUMBNAIL_COLUMN_STATUS not in df.columns:
        df[THUMBNAIL_COLUMN_STATUS] = CrawlStatus.NOT_YET.value
        print(f"Created {THUMBNAIL_COLUMN_STATUS}")

    for i in tqdm(range(len(df))):
        if df[THUMBNAIL_COLUMN_STATUS][i] == CrawlStatus.DONE.value:
            continue

        success = crawler_thumbnail(df["id"][i], category)
        if success:
            df.loc[i, THUMBNAIL_COLUMN_STATUS] = CrawlStatus.DONE.value
        if (i % 100 == 10):
            print("Processing {}".format(i))
            df.to_parquet(path)

        df.to_parquet(path)
        break
