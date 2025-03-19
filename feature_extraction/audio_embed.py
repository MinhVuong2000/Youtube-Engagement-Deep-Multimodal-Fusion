import pandas as pd
import torch
import argparse
import os

def setup_data(entube_path):
    print('Setup data entube')
    return pd.read_parquet(entube_path)

def setup_model():
    print('Setup model VGGish')
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().to(device)

def setup_audio_df(entube, data_sample_dir):
    print('Setup path audio')
    format_path = os.path.join(data_sample_dir, 'audio_by_year', '{}', '{}.wav')
    entube['year_upload'] = entube['upload_date'].dt.year
    audio_df = entube[['id', 'year_upload']]
    audio_df = audio_df[audio_df.year_upload.isin([2016,2017,2020,2021])]
    audio_df['path'] = audio_df.apply(lambda row: format_path.format(row.year_upload, row.id), axis=1)
    return audio_df

def embed_audio_handler(audio_path, model):
    try:
        embed = model.forward(audio_path)
        return embed.tolist()
    except Exception as e:
        print(f'Error at {audio_path}: {e}')
        return None

def embed_audio(audio_df, model, batch_size, start, end, embedded_data_dir):
    print('Start Embedding...')
    audio_embed_path = os.path.join(embedded_data_dir, 'audio_embedding.parquet')
    for idx in range(start, end, batch_size):
        audios_df_batch = audio_df.iloc[idx:idx+batch_size]
        audios_df_batch['embedding_audio'] = audios_df_batch['path'].apply(lambda x: embed_audio_handler(x, model))
        if idx == 0:
            audios_df_batch.to_parquet(audio_embed_path)
        else:
            old_df = pd.read_parquet(audio_embed_path)
            new_df = pd.concat([old_df, audios_df_batch], axis=0)
            new_df.to_parquet(audio_embed_path)
        print(f'Index {idx}->{idx+batch_size} is Done')
    print("All is done")

def main(args):
    entube = setup_data(args.entube_path)
    model = setup_model()
    audio_df = setup_audio_df(entube, args.data_sample_dir)
    embed_audio(audio_df, model, args.batch_size, args.start, args.end, args.embedded_data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio embedding script')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for processing')
    parser.add_argument('--start', type=int, default=0, help='Start index for processing')
    parser.add_argument('--end', type=int, default=None, help='End index for processing')
    parser.add_argument('--entube_path', type=str, required=True, help='Path to ENTUBE_PATH')
    parser.add_argument('--embedded_data_dir', type=str, required=True, help='Path to EMBEDDED_DATA_DIR')
    parser.add_argument('--data_sample_dir', type=str, required=True, help='Path to DATA_SAMPLE_DIR')
    args = parser.parse_args()
    
    if args.end is None:
        args.end = len(pd.read_parquet(args.entube_path))
    
    main(args)
