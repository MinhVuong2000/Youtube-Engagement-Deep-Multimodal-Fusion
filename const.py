#define constants
ROOT_FOLDER = ''
DATA_SAMPLE_DIR = ROOT_FOLDER + 'data_sample/'
EMBEDDED_DATA_DIR = ROOT_FOLDER + 'embedded_data/'
TRAIN_EMBED_PATH = DATA_SAMPLE_DIR+'entube_embedding_train.pt'
VAL_EMBED_PATH = DATA_SAMPLE_DIR+'entube_embedding_val.pt'
TEST_EMBED_PATH = DATA_SAMPLE_DIR+'entube_embedding_test.pt'
ENTUBE_PATH = DATA_SAMPLE_DIR + 'final_entube.parquet'


embedded_video_path = EMBEDDED_DATA_DIR + 'video_embedding.parquet'
embedded_audio_path = EMBEDDED_DATA_DIR + 'audio_embedding.parquet'
embedded_video_path = EMBEDDED_DATA_DIR + 'video_embedding.parquet'
embedded_entube_path = EMBEDDED_DATA_DIR + 'entube_embedding_final.parquet'

SELECT_LABEL = 'label_2'
SAVING_FOLDER = 'trying'
VERSION_MODEL = 'model_main'
CHECKPOINT_DIR = f'{ROOT_FOLDER}checkpoints/{VERSION_MODEL}/{SELECT_LABEL}/{SAVING_FOLDER}'
LOG_DIR = f'{ROOT_FOLDER}logs/{VERSION_MODEL}/{SELECT_LABEL}/{SAVING_FOLDER}'

PATIENCE = 10
BATCH_SIZE = 32
NUM_EPOCH = 30

INPUT_SAME = 512
INPUT_TEXT = 768
INPUT_THUMBNAIL = 2560
INPUT_VIDEO = 2304*2*2
INPUT_AUDIO = 62*128
INPUT_SUM = INPUT_TEXT*2+INPUT_THUMBNAIL+INPUT_VIDEO+INPUT_AUDIO
