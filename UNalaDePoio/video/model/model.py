import cv2
import numpy as np
import os

from ...manager.load_config import LOCAL

def buildModel(weight_path: str = os.path.join(LOCAL, "video", "data", "yolov4-tiny.weights"), cfg_path: str = os.path.join(LOCAL, "video", "data", "yolov4-tiny.cfg")):
    return cv2.dnn.readNet(weight_path, cfg_path)


