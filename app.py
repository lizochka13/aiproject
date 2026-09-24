import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
import time
from mediapipe.tasks.python import vision
import random
import math

#функция которая возвращает сколько пальцев разогнуто
def fingers(dots):
    wrist = dots[0]
    finger_end = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]
    count = 0
    for i in range(4):
        finger = dots[finger_end[i]]
        joint = dots[joints[i]]
        len_end = math.hypot(finger.x - wrist.x, finger.y - wrist.y)
        len_joint = math.hypot(joint.x - wrist.x, joint.y - wrist.y)
        if len_end > len_joint:
            count += 1
    return count
#функция ресайза мемов
def without_distortions(img, target_w, target_h):
    height, width = img.shape[:2]
    coeff = min(target_w/width, target_h/height)
    new_size = [int(height * coeff), int(width * coeff)]
    if coeff < 1:
        resize = cv.resize(img, (new_size[1], new_size[0]), interpolation=cv.INTER_AREA)
    else:
        resize = cv.resize(img, (new_size[1], new_size[0]), interpolation=cv.INTER_CUBIC)
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    padd_h = (target_h - new_size[0]) // 2
    padd_w = (target_w - new_size[1]) // 2
    canvas[padd_h : padd_h + new_size[0], padd_w : padd_w+ new_size[1]] = resize
    return canvas

#скока кадриков держать литсо
NEED_FRAMES = 7 #было 5
#размер окна с вояками
MEME_MIN = 80
#чиселки для подгона порога эмоций
SMILE_MIN = 0.25
JAW_OPEN_MIN  = 0.35
JAW_MIN   = 0.45
BROW_DOWN_MIN  = 0.30
BROW_INNER_MIN = 0.15
FROWN_MIN = 0.10
EYES_UP_MIN   = 0.12
BROW_ASYM_MIN = 0.1
PROFILE_MIN = 0.12

#бери лопату
meme_x = 20
meme_y = 20
meme_size = MEME_MIN

#списочек что бы был ну и типа норм категории блендшейпов видеть удобно
face_list = ["mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight", "browDownLeft", "browDownRight", "browInnerUp", 
             "browOuterUpLeft", "browOuterUpRight", "jawOpen", "eyeSquintLeft", "eyeSquintRight", "tongueOut", "eyeLookUpLeft", 
             "eyeLookUpRight", "eyeBlinkLeft", "eyeBlinkRight"]

#крч модельку тут ищем типа через файлы 
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR /"models" /"face_landmarker.task"
MODEL_PATH2 = BASE_DIR /"models" /"gesture_recognizer.task"
if not MODEL_PATH.exists() or not MODEL_PATH2.exists():
    print("модель не найдена")
    exit() 

#определяем камеру 
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#состояния литса
STATES = ["neutral", "happy", "shock", "angry", "sad", "happy_open", "eyes_up_open", "confused", "profile", "dont_know", "gesture_fist", "point_one", "point_two"]

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
base_options2 = python.BaseOptions(model_asset_path=str(MODEL_PATH2))
recog_options2 = vision.GestureRecognizerOptions(base_options=base_options2, running_mode=vision.RunningMode.VIDEO, num_hands=2)
recognize2 = vision.GestureRecognizer.create_from_options(recog_options2)

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

nose_ratio = 0.5

#основное окно и разворачиваем 
cv.namedWindow('memeface', cv.WINDOW_NORMAL)
cv.resizeWindow('memeface', 900, 700)
while True: #че трешь дурак? дырка будет!
    # читаем кадрик
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #крч у меня камера вебки кринж и я повернула ее и отзеркалила типа можно так то этого не делать если норм будет видно
    frame = cv.flip(frame, 1)
    #frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    #вот проблемы с непонимаем ргб бгр кгб решаются ниже
    frame_for_recognize = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #теперь есть реально объект а не картинка это вобсето важно
    object_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_for_recognize)
    #кручу верчу считаю время в мс
    time_mark = int((time.perf_counter() - start_time) * 1000)
    #посчитали ну теперь получаем результат рекогнайза
    result_recognize = recognize.detect_for_video(object_frame, time_mark)
    result_recognize2 = recognize2.recognize_for_video(object_frame, time_mark)

    calc_state = "" #типа состояние которое нужно посчитать и потом сравнить с кандидатом


    win_x, win_y, win_w, win_h = cv.getWindowImageRect('memeface')


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
        #кончик носа
        nose_center = result_recognize.face_landmarks[0][1].x * w
        #cv.circle(frame, (int(nose_center), int(result_recognize.face_landmarks[0][1].y * h)), 10, (0, 255, 0), -1)
        if head_width != 0:
            nose_ratio = (nose_center - left_head) / head_width
        meme_size = max(MEME_MIN, min(head_width, 300))
        meme_x = left_head - meme_size - 20
        meme_y = up_head
        meme_x = max(0, min(meme_x, w - meme_size))
        meme_y = max(0, min(meme_y, h - meme_size))

    
    #тут делаем жесты с руками пока проверочка на наличие ручек и один жестик его нужно доработать
    if result_recognize2.hand_landmarks:
        if len(result_recognize2.hand_landmarks) == 2:
            if fingers(result_recognize2.hand_landmarks[0]) == 4 and fingers(result_recognize2.hand_landmarks[1]) == 4:
                calc_state = "dont_know"
            if fingers(result_recognize2.hand_landmarks[0]) == 1 and fingers(result_recognize2.hand_landmarks[1]) == 1:
                calc_state = "point_two"
        if len(result_recognize2.hand_landmarks) == 1:
            if fingers(result_recognize2.hand_landmarks[0]) == 0:
                calc_state = "gesture_fist"
            if fingers(result_recognize2.hand_landmarks[0]) == 1:
                calc_state = "point_one"
    #тут проверочка а вообще есть ли лицо и не пустое ли состояние
    if result_recognize.face_blendshapes and calc_state == "":
        # суть то в чем, у вас выходит словарик где ключики это название категорий блендшейпов а значения это скор по каждому
        dict_face_blend = {x.category_name: (0 if x.score - fon.get(x.category_name, 0) < 0 else x.score - fon.get(x.category_name, 0)) for x in result_recognize.face_blendshapes[0]}
        # ну и типа дальше проверочки посчитать че за лицо
        if nose_ratio < PROFILE_MIN or nose_ratio > 1 - PROFILE_MIN:
            calc_state = "profile"
        else:
            if (dict_face_blend["mouthSmileLeft"] + dict_face_blend["mouthSmileRight"])/ 2 > SMILE_MIN and dict_face_blend["jawOpen"] > JAW_OPEN_MIN:
                calc_state = "happy_open"
            elif (dict_face_blend["eyeLookUpLeft"] + dict_face_blend["eyeLookUpRight"])/ 2 > EYES_UP_MIN and dict_face_blend["jawOpen"] > JAW_OPEN_MIN:
                calc_state = "eyes_up_open"
            elif dict_face_blend["jawOpen"] > JAW_MIN: 
                calc_state = "shock"
            elif abs(dict_face_blend["browOuterUpLeft"] - dict_face_blend["browOuterUpRight"]) > BROW_ASYM_MIN:
                calc_state = "confused"
            elif (dict_face_blend["mouthSmileLeft"] + dict_face_blend["mouthSmileRight"])/ 2 > SMILE_MIN:
                calc_state = "happy"
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
        resize_meme = without_distortions(now_img, meme_size, meme_size)
        frame[meme_y : meme_y + meme_size, meme_x : meme_x + meme_size] = resize_meme
        cv.rectangle(frame, (meme_x-2, meme_y-2), (meme_x+meme_size+2, meme_y+meme_size+2), (0,0,0), 2)

    #показываем все
    if win_w > 0 and win_h > 0:
        canvas = without_distortions(frame, win_w, win_h)
        cv.imshow('memeface', canvas)
    else:
        cv.imshow('memeface', frame)

    key = cv.waitKey(1) & 0xFF
    #а тут снчалда считали нажатие а потом зависит от того что нажали делаем разное. типа полезно q - выйти, n - настроить на свое стандарт лицо
    if key == ord('p'):
        if not result_recognize.face_blendshapes:
            print("it's empty")
        else:
            print(nose_ratio)
            #CENA_KOROBKI
            for kluch, value in dict_face_blend.items():
                if kluch in face_list:
                    print(kluch, value)
    if key == ord('f'):
        if not result_recognize2.gestures:
            print("it's empty")
        else:
            for i in range(len(result_recognize2.gestures)):
                hand = result_recognize2.handedness[i][0].category_name
                gesture = result_recognize2.gestures[i][0].category_name
                accuracity = result_recognize2.gestures[i][0].score
                print(hand, gesture, accuracity, fingers(result_recognize2.hand_landmarks[i]))
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