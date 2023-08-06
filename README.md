# Youtube Engagement Analytics via Deep Multimodal Model Fusion
---

The code is implemented for the paper published at Pacific-Rim Symposium on Image and Video Technology 2022 (PSIVT2022),

Read the paper at [here](https://link.springer.com/chapter/10.1007/978-3-031-26431-3_5)

## How to run code:
1. You can get data which is feature extraction at [here](https://drive.google.com/drive/folders/1SM-2VzCQoSAfrI_eGVp8JJ5NU7iKJ6Lx). 
- Data input includes 3 files: entube_embedding_train.pt, entube_embedding_val.pt, entube_embedding_test.pt
- Data in each file is a list with each item is a dictionary including keys:
```python
'id': id of video on Youtube
'embedding_title':tensor which is feature extraction of title, has shape: (768,)
'embedding_tag':tensor which is feature extraction of tag, has shape: (768,)
'embedding_thumbnail':tensor which is feature extraction of thumbnail, has shape: (2560,)
'embedding_video':tensor which is feature extraction of the video, has shape: (2304,1,2,2)
'embedding_audio':tensor which is feature extraction of audio, has shape: (62, 128)
'label_1':tensor of label 1 which not use q-score
'label_2':tensor of label 2 which use q-score
```
2. Clone this repo to your folder and change the current working directory into the folder \
    ```cd <path/to/the/folder>``` 
- You can download and move above data folder like this
    Folder structure:
    ```
    project
    │   README.md
    │   main.py
    │   improved_model.py
    │   const.py
    │   early_stopping.py
    │   multihead_attention.py
    │   requirements.txt
    └───data
        │   entube_embedding_train.pt
        │   entube_embedding_val.pt
        │   entube_embedding_test.pt
    ...
    ```
- If you using colab, you can mount drive, and change path of data in `const.py` file at variables:  `TRAIN_EMBED_PATH`, `VAL_EMBED_PATH`, `TEST_EMBED_PATH`
3. Install neccessary package \
    ```pip install -r requirements.txt```
4. The model when run will have log and checkpoints for each epoch, you can see or change path of them in `const.py` file
5. Run file main.py \
    ```python main.py```
