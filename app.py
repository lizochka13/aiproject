import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import time
from mediapipe.tasks.python import vision
import random
import tkinter as tk

#функция ресайза мемов
def without_distortions(img, side):
    height, width = img.shape[:2]
    coeff = min(side/width, side/height)
    new_size = [int(height * coeff), int(width * coeff)]
    resize = cv.resize(img, (new_size[1], new_size[0]))
    canvas = np.zeros((side, side, 3), np.uint8)
    padd_h = (side - new_size[0]) // 2
    padd_w = (side - new_size[1]) // 2
    canvas[padd_h : padd_h + new_size[0], padd_w : padd_w+ new_size[1]] = resize
    return canvas

#скока кадриков держать литсо
NEED_FRAMES = 7 #было 5
#размер окна с вояками
MEME_MIN = 80
#чиселки для подгона порога эмоций
SMILE_MIN = 0.40
JAW_MIN   = 0.45
BROW_DOWN_MIN  = 0.30
BROW_INNER_MIN = 0.15
FROWN_MIN = 0.10

#бери лопату
meme_x = 20
meme_y = 20

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

#состояния литса
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

#это чеховское ружье обязательно выстрелит
fon = {}

#немного переменных для прикола я не умею в оптимизацию кода ну типа ю ноу шершняга нужно от базы отталкиваться хоть какой-то на смене состояний и картинок
now_state = "neutral"
now_img = None

#кандидат на смену состояний голосуем за него на выборах, и сколько кадров оно держаться должно
candidate = "neutral"
steady = 0 

#основное окно и разворачиваем на весь экран
cv.namedWindow('memeface', cv.WINDOW_NORMAL)
cv.setWindowProperty('memeface', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
#получаем размер окна
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()
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

    calc_state = "" #типа состояние которое нужно посчитать и потом сравнить с кандидатом

    #вот эта куча реально нужна что б мемы за бошкой летали тут все изи находим координаты лица и высчитываем положения мема
    h, w = frame.shape[:2]
    dot_list_x = []
    dot_list_y = []
    if result_recognize.face_landmarks:
        for dots in result_recognize.face_landmarks[0]:
            dot_list_x.append(dots.x)
            dot_list_y.append(dots.y)
        left_head, right_head, up_head, bottom_head = int(min(dot_list_x)*w), int(max(dot_list_x)*w), int(min(dot_list_y)*h), int(max(dot_list_y)*h)
        head_width = right_head - left_head
        meme_size = max(80, head_width)
        meme_x = left_head - meme_size - 20
        meme_y = up_head
        meme_x = max(0, min(meme_x, w - meme_size))
        meme_y = max(0, min(meme_y, h - meme_size))


    #ту проверочка а вообще есть ли лицо
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
        elif (dict_face_blend["browDownLeft"] + dict_face_blend["browDownRight"])/ 2 > BROW_DOWN_MIN:
            calc_state = "angry"
        elif dict_face_blend["browInnerUp"] > BROW_INNER_MIN:
            calc_state = "sad"
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

    #если картиночка имеется мы меняем ее размер впихиваем в основное окно и рамочку делаем
    if now_img is not None:
        resize_meme = without_distortions(now_img, meme_size)
        frame[meme_y : meme_y + meme_size, meme_x : meme_x + meme_size] = resize_meme
        cv.rectangle(frame, (meme_x-2, meme_y-2), (meme_x+meme_size+2, meme_y+meme_size+2), (0,0,0), 2)

    #показываем все
    cv.imshow('memeface', frame)

    key = cv.waitKey(1) & 0xFF
    #а тут снчалда считали нажатие а потом зависит от того что нажали делаем разное. типа полезно q - выйти, n - настроить на свое стандарт лицо
    if key == ord('p'):
        if not result_recognize.face_blendshapes:
            print("it's empty")
        else:
            #CENA_KOROBKI
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
            
cap.release()
cv.destroyAllWindows()