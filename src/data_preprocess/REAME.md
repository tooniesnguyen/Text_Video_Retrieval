# Extract video to keyframe (AutoShot)

## Install
```
pip install -r requirements.txt
```

## Implement
### Step 1: Convert Video to Keyframes
```
cd Autoshot
python implement.py --input_dir <path_to_Keyframes_L0*> \
                    --output_dir <path_to_images>
```
#### Example:
```
python implement.py --input_dir /home/.../data/videos/Keyframes_L02 \
                    --output_dir /home/.../data/images
```

### Step 2: Clean data (check duplicate, dark, blurry, ...)
```
cd clean_data
python clean_vision.py --input_dir <path_to_images>
```
#### Example:
```
python clean_vision.py --input_dir /home/.../data/images/Keyframes_L01
```