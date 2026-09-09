import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import time
from mediapipe.tasks.python import vision
import random

#чиселки для подгона порога эмоций
NEED_FRAMES = 5
SMILE_MIN = 0.40
JAW_MIN   = 0.45
BROW_MIN  = 0.06
FROWN_MIN = 0.10

#списочек что бы был ну и типа норм категории блендшейпов видеть удобно
face_list = ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "browDownLeft", "browDownRight", "browInnerUp", 
             "browOuterUpLeft", "browOuterUpRight", "jawOpen", "eyeSquintLeft", "eyeSquintRight"]

#крч модельку тут ищем типа через файлы 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"models" /"face_landmarker.task"
if not MODEL_PATH.exists():
    print("модель не найдена")
    exit() 

#определяем камеру 
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#состояния ебальничка
STATES = ["neutral", "happy", "shock", "angry", "sad"]

#туть файлики картинок ищем и в словарик кидаем ключ это состояния типа названия папок тоже, а значение это список путей
dir_path = Path(__file__).resolve().parent / "memes" 
statements_dict = { state: list((dir_path / state).glob("*.png")) for state in STATES }
for state, files in statements_dict.items():
    if not files:
        print(f"Ошибка: Папка для состояния '{state}' пуста или не содержит .png файлов! ")


#типа крутые переменные типа я умная знаю ии тут крч просто передаются настройки модельки и сам распознователь
base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
recog_options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces = 1, output_face_blendshapes = True)
recognize = vision.FaceLandmarker.create_from_options(recog_options)

#это вообще кринж какой-то зачем там передавать аргументом время я так и не поняла 
start_time = time.perf_counter()

counter_frames = 0#лол эта строчка вообще нужна?

#то чеховское ружье обязательно выстрелит
fon = {}

#немного переменных для прикола я не умею в оптимизацию кода ну типа ю ноу шершняга нужно от базы отталкиваться хоть какой-то на смене состояний и картинок
now_state = "neutral"
now_img = None

#кандидат на смену состояний голосуем за него на выборах, и сколько кадров оно держаться должно
candidate = "neutral"
steady = 0 

while True: #че трешь дурак? дырка будет!
    # читаем кадрик
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #крч у меня камера вебки кринж и я повернула ее и отзеркалила типа можно так то этого не делать если норм будет видно
    frame = cv.flip(frame, 1)
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    #вот проблемы с непонимаем ргб бгр кгб решаются ниже
    frame_for_recognize = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #теперь есть реально объект а не картинка это вобсето важно
    object_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_for_recognize)
    #кручу верчу считаю время в мс
    time_mark = int((time.perf_counter() - start_time) * 1000)
    #посчитали ну теперь получаем результат рекогнайза
    result_recognize = recognize.detect_for_video(object_frame, time_mark)
    
    counter_frames += 1 #лол эта строчка вообще нужна? часть 2

    calc_state = "" #типа состояние которое нужно посчитать и потом сравнить с кандидатом
    
    if not result_recognize.face_blendshapes:
        calc_state = ""
    else:
        # суть то в чем, у вас выходит словарик где ключики это название категорий блендшейпов а значения это скор по каждому
        dict_face_blend = {x.category_name: (0 if x.score - fon.get(x.category_name, 0) < 0 else x.score - fon.get(x.category_name, 0)) for x in result_recognize.face_blendshapes[0]}
        # ну и типа дальше проверочки посчитать че за лицо
        if (dict_face_blend["mouthSmileLeft"] + dict_face_blend["mouthSmileRight"])/ 2 > SMILE_MIN:
            calc_state = "happy"
        elif dict_face_blend["jawOpen"] > JAW_MIN: 
            calc_state = "shock"
        elif (dict_face_blend["browDownLeft"] + dict_face_blend["browDownRight"])/ 2 > BROW_MIN:
            calc_state = "angry"
        elif (dict_face_blend["mouthFrownLeft"] + dict_face_blend["mouthFrownRight"])/ 2 > FROWN_MIN:
            calc_state = "angry"
        else:
            calc_state = "neutral"
    if calc_state == "":
        pass
    # тут тоже все понятненько типа получается предполагаемое лицо? да +1 к готовности. нет готовность = 1
    elif calc_state == candidate:
        steady += 1
    elif calc_state != candidate:
        candidate = calc_state
        steady = 1
    #и если набрали норм очков готовности это пару секунд лицо подержать то случайную картинку из папки кажем
    if steady >= NEED_FRAMES and candidate != now_state:
        now_state = candidate
        now_img = cv.imread(str(random.choice(statements_dict[now_state])))
    #тут лицо кажем
    cv.imshow('frame', frame)
    key = cv.waitKey(1) & 0xFF
    #а тут снчалда считали нажатие а потом зависит от того что нажали делаем разное. типа полезно q - выйти, n - настроить на свое стандарт лицо
    if key == ord('p'):
        if not result_recognize.face_blendshapes:
            print("it's empty")
        else:
            for kluch, value in dict_face_blend.items():
                if kluch in face_list:
                    print(kluch, value)
    if key == ord('n'):
            if not result_recognize.face_blendshapes:
                print("it's empty")
            else:
                #БААААААМ!
                fon = {x.category_name: x.score for x in result_recognize.face_blendshapes[0]}
                print(fon)
    if key == ord('q'):
            break
            
    #мемы, проходите в окошко номер 2
    if now_img is not None:
        cv.imshow('meme', now_img)
cap.release()
cv.destroyAllWindows()