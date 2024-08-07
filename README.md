# Text_video_Retrieval
AI Challenge 2024




## Tree folder

```
|---data
|   |--images
|   │   |--Keyframes_L01
|   |   |       |--L01_V001
|   |   |               |-- *.jpg
|   |   |       |--L01_V002
|   │   |--Keyframes_L02
|   
|   |--videos
|       |--Keyframes_L01
|           |--*.mp4
|
|---dicts
|   |--data_ocr
|           |--Keyframes_L01
|                   |--Keyframes_L01.txt
|                   |--L01_V00*.txt
|           |--Keyframes_L02
|   |--info_ocr.txt
|   |--*.json
|
|---model
|   |--ocr_feats
|       |--npy/*.npy
|       |--bin/*.bin
|
|   |--asr_feats
|
|   |--imgcap_feat
|
|
|--src
|
|---README.md
```

# Tutorial

## Create fake dataset
- Move to `data` folder
```
cd data 
bash get_data.sh
```
- Sau khi chạy lệnh xong sẽ có folder video theo như cấu trúc cây thư mục trên
  
## [Extract video to keyframe (AutoShot)](./src/data_preprocess/REAME.md)
- Sau khi chạy xong phần này mn sẽ nhận được một bộ cấu trúc tập data giả được phân bố theo cây thư mục giống lúc thi

# References
- https://github.com/Syun1208/text-video-retrieval
- https://github.com/AIVIETNAMResearch/Video-Text-Retrieval