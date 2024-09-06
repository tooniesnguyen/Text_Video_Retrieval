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


HOW TO USE?


First run "F:\Text_Video_Retrieval\utils\ocr_processing\OCR_Extract.py" 

by using this code:

python OCR_Extract.py --root_path [path root of data] --save_path [path to save] (Code just rn with one GPU)

than run code to combine all file txt:
python Combine_file.py --root_dir [root txt] --output_file [path to save]

example :
python Combine_file.py --root_dir "/content/Ocr/" --output_file "/content/info_ocr.txt"

next, run:

python Ocr2Npy.py --ocr_inf [path of ocr file] --ocr_save_np [path to save npy] --combine  


example:
python /content/Ocr_for_AIC/utils/ocr_processing/Ocr2Npy.py --ocr_inf /content/info_ocr.txt --ocr_save_np /content/npy2 --combine 

Finally:

Custom link in OcrBin.py function and run



# References
- https://github.com/Syun1208/text-video-retrieval
- https://github.com/AIVIETNAMResearch/Video-Text-Retrieval
