# Extract video to keyframe (AutoShot)

## Install
```
pip install -r requirements.txt
```

## Implement
### Step 1: Convert Video to Keyframes
```
cd Autoshot
python implement.py --i <path_to_Keyframes_L0*> \
                    --o <path_to_images>
```
#### Example:
```
python implement.py --i /home/.../data/videos/Keyframes_L02 \
                    --o /home/.../data/images
```

### Step 2: Clean data (check duplicate, dark, blurry, ...)
```
cd clean_data
python clean_vision.py --i <path_to_images>
```
#### Example:
```
python clean_vision.py --i /home/.../data/images/Keyframes_L01
```