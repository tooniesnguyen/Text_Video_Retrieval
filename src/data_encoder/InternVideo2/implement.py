import numpy as np
import os
import io
import cv2

import torch

from utils.config_impl import (Config,
                               eval_dict_leaf)

from utils.utils_impl import (retrieve_text,
                              _frame_from_video,
                              setup_internvideo2)

print("Hello World")