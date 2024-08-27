# Create .npy and .bin file (InternVideo2)

## Installation

Please follow the installation instructions in [INSTALL](./INSTALL.md).

>The codebase support using [wandb](https://wandb.ai/) to monitor training. If you want to use wandb, you will need to set up it following [this very short instruction](https://docs.wandb.ai/quickstart#1.-set-up-wandb), and also set `wandb.enable` in the config to be `True`. `wandb.entity` and `wandb.project` should also be set.


## CLIP Post-pretraining

We use the text encoder from [InternVL-CLIP](https://github.com/OpenGVLab/InternVL/tree/main/clip_benchmark).


**Set up step:**
1. Download [InternVideo2-stage2_1b-224p-f4.pt](https://huggingface.co/OpenGVLab/InternVideo2/blob/main/InternVideo2-stage2_1b-224p-f4.pt) 
2. Put this file in path: `Text_Video_Retrieval/model/internvideo2_feats/Internvideo2/weights`


## Create .npy and .bin file
:warning: **data_file_path structure:**
```$ tree -d
|---data
   |--images
   │   |--Keyframes_L01
   |   |       |--L01_V001
   |   |       |       |-- *.jpg
   |   |       |--L01_V001.txt
   |   |       |--L01_V002
   |   |       |       |-- *.jpg
   |   |       |--L01_V0012.txt
   │   |--Keyframes_L02
   |
   |--videos
       |--Keyframes_L01
           |--*.mp4
```
**Create .npy files:**
```shell
cd Text_Video_Retrieval/model/internvideo2_feats/Internvideo2
python internVideo2_feats.py  --img2npy -i [data_file_path] -o [output_npy_path]
```

**Create .bin files:**
```shell
cd Text_Video_Retrieval/model/internvideo2_feats/Internvideo2
python internVideo2_feats.py  --npy2bin -i [*.npy_path_input] -o [output_bin_path]
```
:warning: *npy_path_input structure:*
```$ tree -d
|---npy_path_input
   |--L01_V001.npy
   |--L01_V002.npy
   |--L02_V001.npy
   |--L02_V002.npy
   |--*.npy

```


