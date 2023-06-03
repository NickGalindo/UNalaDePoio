import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

from ...manager.load_config import LOCAL

import os

import numpy as np

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

TEXT_COLOR = (255, 0, 0)

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(LOCAL, "video", "data", "efficientdet_lite2.tflite")),
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

detector = ObjectDetector.create_from_options(options)

def predictImg(idx, img, out_path):
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    preds = detector.detect(rgb_frame)

    rgb_frame = np.copy(rgb_frame.numpy_view())

    cat = "OTROS"
    prob = 0

    for i in preds.detections:
        bbox = i.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(rgb_frame, start_point, end_point, TEXT_COLOR, 3)

        category = i.categories[0]
        if category.score > prob:
            prob = category.score
            cat = category.category_name

    mod_img = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    img_path = os.path.join(out_path, f"{idx}.jpg")
    cv2.imwrite(img_path, mod_img)

    return cat

