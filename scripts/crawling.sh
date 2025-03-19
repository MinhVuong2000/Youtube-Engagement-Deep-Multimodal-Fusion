#!/bin/bash
export PYTHONPATH="/Users/vuongnguyen/Downloads/Github/Youtube-Engagement-Deep-Multimodal-Fusion:$PYTHONPATH"
for category in train val test; do
    cd crawl
    python title_tags.py --category $category
    python thumbnail.py --category $category
    python video.py --category $category
    python audio.py --category $category
done
