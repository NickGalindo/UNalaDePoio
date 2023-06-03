import cv2
from ..predict.predict import predictImg
from PIL import Image
import easyocr
import pandas as pd
import os 
import time

reader = easyocr.Reader(['es'])
obj_map = {"fire hydrant": "CONSTRUCCION", "stop sign": "VIA", "car": "VEHICULO", "bicycle": "VEHICULO", "motorcycle": "VEHICULO", "airplane": "VEHICULO", "train": "VEHICULO", "truck": "VEHICULO", "boat": "VEHICULO"}

def splitVideo(path: str, out_path: str):
    capture = cv2.VideoCapture(path)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
    # With webcam get(CV_CAP_PROP_FPS) does not work.
    # Let's see for ourselves.
 
    if int(major_ver)  < 3 :
        fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else :
        fps = capture.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    num_frames = 0

    info = {"ID": [], "OBJECT_TYPE": [], "TIME": [], "COORDINATES_TEXT": []}

    frame_freq = 30
    cur_frame = 0
    idx = 0
    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            break

        num_frames += 1
        cur_frame += 1

        if cur_frame == frame_freq:
            cur_frame = 0
            idx += 1

            txt_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text = reader.readtext(txt_frame)

            new_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
            cat = predictImg(idx, new_frame, os.path.join(os.path.dirname(out_path), "IMG"))

            ti = (num_frames/fps)*1000
            ti = time.strftime('%H:%M:%S', time.gmtime(ti))

            text = " ".join([rd[1] for rd in text])

            cat = obj_map[cat] if cat in obj_map else "OTROS"

            res = {"ID": idx, "OBJECT_TYPE": cat, "TIME": ti, "COORDINATES_TEXT": text}

            for key in info:
                info[key].append(res[key])

            print(res)

    capture.release()

    df = pd.DataFrame(info)

    df.to_csv(out_path)
