from enum import Enum
import pandas as pd


class CrawlStatus(Enum):
    NOT_YET="NOT_YET"
    DONE="DONE"


def load_data(category, path_sample='data/{category}.parquet'):
    path = path_sample.format(category=category)
    df = pd.read_parquet(path)
    return df

TEXT_COLUMN_STATUS = "title_tag_crawl_status"
THUMBNAIL_COLUMN_STATUS = "thumbnail_crawl_status"
VIDEO_AUDIO_COLUMN_STATUS = "crawl_status" # "video_audio_crawl_status"
