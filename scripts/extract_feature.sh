#!/bin/bash
export PYTHONPATH="/Users/vuongnguyen/Downloads/Github/Youtube-Engagement-Deep-Multimodal-Fusion:$PYTHONPATH"

for dataset in train test val; do
    python feature_extraction/thumnail_embed.py --entube_path data/entube_${dataset}.parquet --embedded_data_dir data/embedded_data --data_sample_dir data/samples --start 0 --end None --batch_size 100
    python feature_extraction/video_embed.py --entube_path data/entube_${dataset}.parquet --embedded_data_dir data/embedded_data --data_sample_dir data/samples --start 0 --end None --batch_size 100
    python feature_extraction/audio_embed.py --entube_path data/entube_${dataset}.parquet --embedded_data_dir data/embedded_data --data_sample_dir data/samples --start 0 --end None --batch_size 100
    python feature_extraction/title_tag_embed.py --entube_path data/entube_${dataset}.parquet --embedded_data_dir data/embedded_data --data_sample_dir data/samples --start 0 --end None --batch_size 100
done
