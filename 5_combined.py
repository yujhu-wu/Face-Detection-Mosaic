import cv2
import tensorflow as tf
import numpy as np

# 載入人臉辨識模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('face.yml')

# 載入人臉追蹤模型
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 載入口罩辨識模型
model = tf.keras.models.load_model('keras_model.h5', compile=False)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def text_msg(text_msg):      # 建立顯示文字的函式
    global img
    org = (0, 50)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    lineType = cv2.LINE_AA
    cv2.putText(img, text_msg, org, fontFace, fontScale, color, thickness, lineType)


name = {
    '1': '[25]YU-JHU',
    '2': '[27]YUN-JHEN',
    '3': '[41]YI-JHEN',
    '4': '[46]TZU-LING'
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

mosaic_applied = True  # 標記是否已應用馬賽克
text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break

    img = cv2.resize(frame, (810, 450))           # 固定尺寸
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換成灰階
    faces = face_cascade.detectMultiScale(gray)   # 追蹤人臉

# 顯示口罩判斷結果和應用馬賽克
    # 進行口罩辨識
    crop_x = int((img.shape[1] - 320) / 2)
    crop_y = int((img.shape[0] - 240) / 2)
    cropped_img = img[crop_y:crop_y+240, crop_x:crop_x+320]
    resized_img = cv2.resize(cropped_img, (224, 224))
    image_array = resized_img.astype(np.float32) / 127.0 - 1
    data[0] = image_array
    prediction = model.predict(data)
    a,b,bg= prediction[0]
    print(a,b,bg)
    if a> 0.699:    # 判斷沒戴口罩
        text_msg('No mask~')
    if b>0.301:     # 判斷有戴口罩
        text_msg('Yes mask,please take it off!')


    # 應用馬賽克
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 標記人臉外框

        # 進行人臉辨識
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 65:
            text = name[str(idnum)]  # 如果信心指數小於 65，取得對應的名字
        else:
            text = '???'  # 否則名字就是'???'

        cv2.putText(img, text, (x,y-5),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        if mosaic_applied:
            mosaic = img[y:y + h, x:x + w]
            level = 15
            mh = int(h / level)
            mw = int(w / level)
            mosaic = cv2.resize(mosaic, (mw, mh), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
            img[y:y + h, x:x + w] = mosaic


        key = cv2.waitKey(1)
        if text == '[25]YU-JHU' and key == ord('1'):        # 按下數字鍵'1'
            mosaic_applied = not mosaic_applied                 # 切換[25號-雨茿]馬賽克應用狀態
        elif text == '[27]YUN-JHEN' and key == ord('2'):    # 按下數字鍵'2'
            mosaic_applied = not mosaic_applied                 # 切換27號-芸禎]馬賽克應用狀態
        elif text == '[41]YI-JHEN' and key == ord('3'):     # 按下數字鍵'3'
            mosaic_applied = not mosaic_applied                 # 切換[41號-奕臻]馬賽克應用狀態
        elif text == '[46]TZU-LING' and key == ord('4'):    # 按下數字鍵'4'
            mosaic_applied = not mosaic_applied                 # 切換[46號-子玲]馬賽克應用狀態

    cv2.imshow('Mask + Camera Mosaic', img)
    if cv2.waitKey(1) == ord('q'):
        break    # 按下'q'鍵停止

cap.release()
cv2.destroyAllWindows()
