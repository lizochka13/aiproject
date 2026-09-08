import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import time
from mediapipe.tasks.python import vision
import random

face_list = ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "browDownLeft", "browDownRight", "browInnerUp", 
             "browOuterUpLeft", "browOuterUpRight", "jawOpen", "eyeSquintLeft", "eyeSquintRight"]

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"models" /"face_landmarker.task"
if not MODEL_PATH.exists():
    print("модель не найдена")
    exit() 
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


list_path_neutral = []
list_path_happy = []
dir_path = Path(__file__).resolve().parent
dir_path = dir_path / "memes"
for file_path in dir_path.glob('**/*'):
    if file_path.is_file(): 
        if "neutral" in str(file_path):
            list_path_neutral.append(file_path)
        if "happy" in str(file_path):
            list_path_happy.append(file_path)
statements_dict = {"neutral":list_path_neutral, "happy":list_path_happy}


base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
recog_options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces = 1, output_face_blendshapes = True)
recognize = vision.FaceLandmarker.create_from_options(recog_options)

start_time = time.perf_counter()
counter_frames = 0

fon = {}

now_state = "neutral"
now_img = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame, 1)
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    frame_for_recognize = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    object_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_for_recognize)
    time_mark = int((time.perf_counter() - start_time) * 1000)
    result_recognize = recognize.detect_for_video(object_frame, time_mark)
    
    counter_frames += 1

    calc_state = ""
    
    if not result_recognize.face_blendshapes:
        calc_state = ""
    else:
        dict_face_blend = {x.category_name: (0 if x.score - fon.get(x.category_name, 0) < 0 else x.score - fon.get(x.category_name, 0)) for x in result_recognize.face_blendshapes[0]}
        if (dict_face_blend["mouthSmileLeft"] + dict_face_blend["mouthSmileRight"])/ 2 > 0.4:
            calc_state = "happy"
        else: 
            calc_state = "neutral"

    cv.imshow('frame', frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('p'):
        if not result_recognize.face_blendshapes:
            print("it's empty")
        else:
            dict_face_blend = {x.category_name: x.score for x in result_recognize.face_blendshapes[0]}
            print(sorted(dict_face_blend.items(), key=lambda item: item[1], reverse=True))
    if key == ord('n'):
            fon = {x.category_name: x.score for x in result_recognize.face_blendshapes[0]}
            print(fon)
    if key == ord('q'):
            break
    if calc_state:
        if calc_state != now_state:
            now_state = calc_state
            now_img = cv.imread(str(random.choice(statements_dict[now_state])))
    
    if now_img is not None:
        cv.imshow('meme', now_img)
cap.release()
cv.destroyAllWindows()